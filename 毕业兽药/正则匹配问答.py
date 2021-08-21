import re

pattern = [
           [r"^.\w+颁布了哪些食品安全法规？", r"^.\w+出台了哪些食品安全法规？"],
           [r"\w+主题的法规有哪些", r"关于\w+的法规有哪些"],
          ]
question = '河南省主题的法规有哪些'
pos = -1
q_type = -1
for i in range(len(pattern)):
    for x in pattern[i]:
        index = re.search(x, question)
        if (index):
            pos = index.span()[0]
            q_type = i
            break
    if (pos != -1):
        break

print(pos)