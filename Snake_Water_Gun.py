import random
import time
i=1
win=0
loose=0
print("There will be total 3 round based on that winner will be decided\n\n")
time.sleep(1)
while(True):
    print("Round", i)
    user_input = input("What you wanna choose\n"
              "For Snake choose Press S\n"
              "For Water choose W\n"
              "For Gun choose G\n\t")
    y=user_input.upper()
    a = ["S", "W", "G"]
    x = random.choice(a)
    i=i+1
    if x == y:
        print("system had choosen  : ", x, "and your choice was ", y)
        print("Game is tie\n\n")
    elif x=="S" :
        if y=="W":
            print("system had choosen  : ", x, "and your choice was ", y)
            print("You loose\n\n")
            loose=loose+1
            print("your total loose is ", loose)
        else :
            print("system had choosen  : ", x, "and your choice was ", y)
            print("You win\n\n")
            win=win+1
            print("your total win is : ",win)
    elif x=="W":
        if y=="S":
            print("system had choosen  : ", x, "and your choice was ", y)
            print("You win")
            win = win + 1
            print("your total win is : ", win)
        else :
            print("system had choosen  : ", x, "and your choice was ", y)
            print("You loose\n\n")
            loose = loose + 1
            print("your total loose is ", loose)
    else :
        if y=="S":
            print("system had choosen  : ", x, "and your choice was ", y)
            print("You loose\n\n")
            loose = loose + 1
            print("your total loose is ", loose)
        else:
            print("system had choosen  : ", x, "and your choice was ", y)
            print("You win")
            win = win + 1
            print("your total win is : ", win)

    if i==4:
        print("Game Complete\n")
        print("Total win ", win,"Total loose",loose)
        final_score=int(win)-int(loose)
        print("final score",final_score)
        if final_score<0:
            print("you have lost the game")
        elif final_score==0:
            print("Game is tie")
        else:
            print("congratulation you win")
        break


