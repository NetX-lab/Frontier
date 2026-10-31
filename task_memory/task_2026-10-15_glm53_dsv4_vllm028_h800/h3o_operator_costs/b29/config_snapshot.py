import json,sys
from pathlib import Path
r=Path(sys.argv[1]);case=sys.argv[2];directory=Path(sys.argv[3]);out=Path(sys.argv[4])
sys.path.insert(0,str(r/'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10'))
import a10_driver
fields={'trace_request_generator_config_trace_file':str(Path(__file__).resolve().parent/'trace_first8.csv')}
if hasattr(a10_driver,'REQUEST_INGRESS_DELAY_MS'):
 fields.update(request_ingress_delay_ms=0.0,analytical_kv_cache_transfer_config_request_handoff_delay_ms=0.0)
config,changes=a10_driver.build_config(case,directory,fields)
out.write_text(json.dumps(dict(flat=config.to_dict(),changes=changes),indent=2,default=str)+'\n')
