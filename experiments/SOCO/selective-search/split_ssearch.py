import json
res_file = 'imagenet1k_ssearch_train.json'
out_dir = 'ssearch_imagenet/'
with open(res_file) as f:
    lines = f.readlines()
for line in lines:
    try:
        info = json.loads(line)
    except:
        continue
    if len(info['instances']) == 0:
        continue 
    name = info['filename'].split('.')[0]
    name = name.split('/')[0] + '-' + name.split('/')[1]
    f_write = open(out_dir+name+'.txt', 'w')
    f_write.write(json.dumps(info) + "\n")
    f_write.close()
