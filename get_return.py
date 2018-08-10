import csv

def get_return(filename,title_list):
  with open(filename) as f:
    reader = csv.DictReader(f)
    return_list = []
    for row in reader:
      row_data=[]
      for title in title_list:
        row_data.append(row[title])
      return_list.append(row_data)
    return return_list

headers = ['rollout/return','eval/return']
ddpg = get_return('ddpglog/progress.csv',headers)
meta = get_return('meta_log/progress.csv',headers)

assert len(ddpg)>=len(meta)

for index in range(len(meta)):
  ddpg[index].extend(meta[index])
  
new_headers=[]
for pre in ['ddpg/','meta/']:
  for name in headers:
    new_headers.append(pre+name)

with open('return.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerow(new_headers)
  for row in ddpg:
    writer.writerow(row)
