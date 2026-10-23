#!/usr/bin/env python3
from pathlib import Path
import re,sys,math,json
root=Path(sys.argv[1])
results=[]
for name in ('mode1','mode2','cold'):
 text=(root/(name+'.log')).read_text()
 assert 'HODLR GPU factor (' in text
 assert 'device_factor_bytes_local=' in text, name+': missing factor bytes'
 factors=re.findall(r'device_factor_bytes_local=(\S+)',text)
 assert factors and max(map(float,factors))>0
 assert 'device_used_bytes_all_processes=' in text, name+': missing device memory'
 values=re.findall(r'operation=inverse.*?rhs_boundary_bytes_local=(\d+)',text)
 assert values and max(map(int,values))>0
 diffs=re.findall(r'relative difference to the CPU\s+(\S+)',text)
 assert diffs and all(math.isfinite(float(v)) and float(v)<1.e-9 for v in diffs)
 results.append(dict(case=name,passed=True,max_gpu_cpu_difference=max(map(float,diffs)),rhs_bytes=max(map(int,values))))
for name in ('mode0','cpu1','cpu2'):
 text=(root/(name+'.log')).read_text()
 assert 'HODLR hybrid interface: PASS' in text
 assert 'backend=GPU' not in text
 assert 'construction=host_callbacks_plus_upload' not in text
(root/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
print('Focused hybrid interface checks: PASS')
