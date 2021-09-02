def preprocess_transcript_bznsyp(dict_info, dict_transcript):
    transList = []
    for t in dict_transcript:
        transList.append(t)
    for i in range(0, len(transList), 2):
        if not transList[i]:
            continue
        key = transList[i].split("\t")[0]
        transcript = transList[i+1].strip().replace("\n","").replace("\t"," ")
        dict_info[key] = transcript