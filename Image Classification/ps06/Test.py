def get_num_from_text(str1):
    idx2 = str1.index('.')
    return int(str1[7:idx2:1])


print(get_num_from_text("subject01.centerlight.png"))