import json
import datetime

fn='/home/ubuntu/report_request.file/report_request.js'

cur_dt = datetime.datetime.now()

with open(fn) as f:

   d = json.load(f)
   d['report']['start_date'] = str(cur_dt - datetime.timedelta(days=8))[:10] + ' 00:00:00'
   d['report']['end_date'] = str(cur_dt)[:10] + ' 00:00:00'

with open(fn, 'w') as f:
   json.dump(d, f)
