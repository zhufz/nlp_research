#-*- coding:utf-8 -*-
import pdb
from collections import defaultdict

class node(object):

    def __init__(self):
        self.next = {}
        self.fail = None
        self.isWord = False
        self.word = ""

class AC(object):

    def __init__(self):
        self.root = node()

    def add(self, word):
        temp_root = self.root
        for char in word:
            if char not in temp_root.next:
                temp_root.next[char] = node()
            temp_root = temp_root.next[char]
        temp_root.isWord = True
        temp_root.word = word

    def make_fail(self):
        temp_que = []
        temp_que.append(self.root)
        while len(temp_que) != 0:
            temp = temp_que.pop(0)
            p = None
            for key,value in temp.next.item():
                if temp == self.root:
                    temp.next[key].fail = self.root
                else:
                    p = temp.fail
                    while p is not None:
                        if key in p.next:
                            temp.next[key].fail = p.fail
                            break
                        p = p.fail
                    if p is None:
                        temp.next[key].fail = self.root
                temp_que.append(temp.next[key])

    def search(self, content):
        p = self.root
        result = defaultdict(list)
        currentposition = 0

        #pdb.set_trace()
        while currentposition < len(content):
            word = content[currentposition]
            #print(word)
            while word in p.next == False and p != self.root:
                p = p.fail

            if word in p.next:
                p = p.next[word]
            else:
                p = self.root
                if word in p.next:
                    p = p.next[word]

            if p.isWord:
                #pdb.set_trace()
                #print(">>>>>>",p.word)
                result[p.word].append((currentposition-len(p.word)+1,currentposition+1))

            currentposition += 1
        return dict(result)

if __name__ == '__main__':
    ac = AC()
    ac.add('指示灯')
    ac.add('垃圾')
    ac.add('生存游戏')
    res = ac.search('打开指示灯')
    print(res)
    #{'指示灯': [(2, 5)]}
