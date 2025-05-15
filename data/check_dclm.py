from datatrove.pipeline.readers import ParquetReader

data_dclm = ParquetReader("../xxxx/datasets/dclm-baseline-1.0-parquet", read_metadata=True)()
cnt_d = {}
for d in data_dclm:
    url = d.metadata["url"]
    url = url.replace("http://", "").replace("https://", "")
    if url[0] not in cnt_d:
        cnt_d[url[0]] = 0
    cnt_d[url[0]] += 1

print(cnt_d)
