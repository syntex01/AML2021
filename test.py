import data
id_count=10
balance_classes=True
a = data.Data()
data = a.data

# scan all not negative images for missing boxes
data_not_negative = data[data["negative"] != 1]
blacklist = data_not_negative[data_not_negative["boxes"].str.len() == 0]["image_id"]

# delete the rows containing missing boxes from the dataframe (~300)
data_cleaned = data[~data["image_id"].isin(blacklist)]

sampled_ids = []
if balance_classes:
    count_per_class = int(id_count / 4)
    for j in data.columns.values[5:9]:
        sampled_ids.extend(list(data_cleaned['image_id'][data[j] == 1].sample(n=count_per_class)))
    return sampled_ids
else:
    return list(data_cleaned['image_id'].sample(n=id_count))