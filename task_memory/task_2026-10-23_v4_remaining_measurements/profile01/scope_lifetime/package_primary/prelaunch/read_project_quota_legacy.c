#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/dqblk_xfs.h>
#include <stdio.h>
#include <string.h>
#include <sys/quota.h>
#include <sys/syscall.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    int fd = open(argv[1], O_RDONLY | O_DIRECTORY);
    if (fd < 0) { perror("open filesystem"); return 3; }
    struct fs_quota_statv state = { .qs_version = FS_QSTATV_VERSION1 };
    if (quotactl(QCMD(Q_XGETQSTATV, PRJQUOTA), "/dev/rbd1", 0, (void *)&state)) {
        fprintf(stderr, "Q_XGETQSTATV errno=%d: %s\n", errno, strerror(errno));
        return 4;
    }
    fs_disk_quota_t quota = { .d_version = FS_DQUOT_VERSION, .d_flags = FS_PROJ_QUOTA };
    if (quotactl(QCMD(Q_XGETQUOTA, PRJQUOTA), "/dev/rbd1", 0, (void *)&quota)) {
        fprintf(stderr, "Q_XGETQUOTA project=0 errno=%d: %s\n", errno, strerror(errno));
        return 5;
    }
    printf("{\"quota_flags\":%u,\"project_accounting\":%s,\"project_enforcement\":%s,"
           "\"project_id\":%u,\"block_unit_bytes\":512,\"blocks_used\":%" PRIu64 ","
           "\"blocks_hardlimit\":%" PRIu64 ",\"blocks_softlimit\":%" PRIu64 ","
           "\"inodes_used\":%" PRIu64 ",\"inodes_hardlimit\":%" PRIu64 "}\n",
           state.qs_flags, (state.qs_flags & FS_QUOTA_PDQ_ACCT) ? "true" : "false",
           (state.qs_flags & FS_QUOTA_PDQ_ENFD) ? "true" : "false", quota.d_id,
           (uint64_t)quota.d_bcount, (uint64_t)quota.d_blk_hardlimit,
           (uint64_t)quota.d_blk_softlimit, (uint64_t)quota.d_icount,
           (uint64_t)quota.d_ino_hardlimit);
    close(fd);
    return 0;
}
