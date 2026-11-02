"""Read and verify local Git objects, including packed objects, without Git commands."""
from functools import lru_cache
import hashlib
from pathlib import Path
import struct
import zlib

OBJECTS=Path('/data/ycfeng/Frontier/.git/objects')
TYPES={1:b'commit',2:b'tree',3:b'blob',4:b'tag'}


def pack_location(oid):
    key=bytes.fromhex(oid)
    for index in sorted((OBJECTS/'pack').glob('*.idx')):
        with index.open('rb') as stream:
            assert stream.read(8)==b'\xfftOc\x00\x00\x00\x02', index
            counts=struct.unpack('>256I',stream.read(1024));count=counts[-1]
            low=counts[key[0]-1] if key[0] else 0;high=counts[key[0]]
            while low<high:
                mid=(low+high)//2;stream.seek(1032+20*mid);candidate=stream.read(20)
                if candidate<key:low=mid+1
                else:high=mid
            if low==count:continue
            stream.seek(1032+20*low)
            if stream.read(20)!=key:continue
            stream.seek(1032+24*count+4*low);offset=struct.unpack('>I',stream.read(4))[0]
            if offset&0x80000000:
                stream.seek(1032+28*count+8*(offset&0x7fffffff));offset=struct.unpack('>Q',stream.read(8))[0]
            return index.with_suffix('.pack'),offset
    raise FileNotFoundError('Git object absent from local loose and packed stores: '+oid)


def varint(data,position):
    value=shift=0
    while True:
        byte=data[position];position+=1;value|=(byte&127)<<shift
        if not byte&128:return value,position
        shift+=7


def apply_delta(base,delta):
    base_size,pos=varint(delta,0);result_size,pos=varint(delta,pos)
    assert len(base)==base_size
    result=bytearray()
    while pos<len(delta):
        opcode=delta[pos];pos+=1
        if opcode&128:
            offset=size=0
            for bit in range(4):
                if opcode&(1<<bit):offset|=delta[pos]<<(8*bit);pos+=1
            for bit in range(3):
                if opcode&(16<<bit):size|=delta[pos]<<(8*bit);pos+=1
            size=size or 0x10000
            assert offset+size<=len(base)
            result.extend(base[offset:offset+size])
        else:
            assert opcode and pos+opcode<=len(delta)
            result.extend(delta[pos:pos+opcode]);pos+=opcode
    assert len(result)==result_size
    return bytes(result)


@lru_cache(maxsize=128)
def packed_object(path,offset):
    with path.open('rb') as stream:
        assert stream.read(4)==b'PACK'
        stream.seek(offset);first=stream.read(1)[0];kind=(first>>4)&7;size=first&15;shift=4;byte=first
        while byte&128:
            byte=stream.read(1)[0];size|=(byte&127)<<shift;shift+=7
        base_offset=base_oid=None
        if kind==6:
            byte=stream.read(1)[0];distance=byte&127
            while byte&128:
                byte=stream.read(1)[0];distance=((distance+1)<<7)+(byte&127)
            base_offset=offset-distance
        elif kind==7:base_oid=stream.read(20).hex()
        decoder=zlib.decompressobj();chunks=[]
        while not decoder.eof:
            block=stream.read(65536);assert block, (path,offset)
            chunks.append(decoder.decompress(block))
        data=b''.join(chunks);assert len(data)==size
    if kind in TYPES:return TYPES[kind],data
    assert kind in (6,7)
    base_kind,base=packed_object(path,base_offset) if kind==6 else raw_object(base_oid)
    return base_kind,apply_delta(base,data)


@lru_cache(maxsize=128)
def raw_object(oid):
    path=OBJECTS/oid[:2]/oid[2:]
    if path.exists():
        raw=zlib.decompress(path.read_bytes());header,data=raw.split(b'\0',1);kind,size=header.split(b' ',1)
        assert len(data)==int(size)
    else:
        path,offset=pack_location(oid);kind,data=packed_object(path,offset)
        raw=kind+b' '+str(len(data)).encode()+b'\0'+data
    assert hashlib.sha1(raw).hexdigest()==oid,oid
    return kind,data


def object_data(oid):
    return raw_object(oid)[1]
