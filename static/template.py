head = """import sys
import pickle
import copy
res = {}
code_len = %d
cnt = 0
last_line = [None]
def trace_func_local_vars(frame, event, arg):
    if event not in ("line", "return"):
        return trace_func_local_vars
    global cnt
    _locals = {}
    for x, y in frame.f_locals.items():
        if not str(y).startswith('<'):
            if type(y) in [set, range]:
                _locals[x] = list(y)
            else:
                _locals[x] = y
    if event == "line":
        line = frame.f_lineno - 36
        if 0 < line <= code_len:
            cnt += 1
            res.setdefault("lines_seq", []).append(line)
            if not line in res:
                # 此处存在浅拷贝导致的问题，因此要用deepcopy
                res[line] = {'start': copy.deepcopy(_locals), 'end': {}}
            if last_line[0] in res:
                res[last_line[0]]['end'] = copy.deepcopy(_locals)
            res[line]["index"] = cnt
            last_line[0] = line
    elif event == "return":
        if last_line[0] in res and not res[last_line[0]]['end']:
            res[last_line[0]]['end'] = copy.deepcopy(_locals)
    return trace_func_local_vars
sys.settrace(trace_func_local_vars)"""



tail = """
sys.settrace(None)
print()
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
tmp_dir = PROJECT_ROOT / "tmp"
tmp_dir.mkdir(parents=True, exist_ok=True)

with open(tmp_dir / "trace_tmp.pkl", 'wb') as f:
    pickle.dump(res, f)
with open(tmp_dir / "trace_tmp.json", 'w') as f:
    import json
    json.dump(res, f, indent=4)
"""


tmp = """
# if not str(frame.f_code.co_name) in ["mock_input", "<module>", "<lambda>", "func",]
# stack = traceback.extract_stack(limit=2)
# code = traceback.format_list(stack)[0].split("\n")[1].strip()
# res[str(line)].append({
#     "Event": event,
#     "Func": str(frame.f_code.co_name),
#     "Line": ,
#     "raw_code": code,
#     "local_vars": str(_locals)
# })
# print(
#     "Event: {0}  Func: {1}, Line: {2}, raw_code: {3}, local_vars: {4}".format(
#         event, frame.f_code.co_name, frame.f_lineno - 29, code, _locals
#     )
# )
"""


# for print or logging
print_spliter = '-' * 20 + ' {} ' + '-' * 20