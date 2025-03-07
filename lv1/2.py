
try:
    print("Upišite broj: ")
    broj = float(input()) 

    if(broj < 0.0 or broj > 1.0):   
        print("Broj nije u intervalu [0.0 i 1.0]! Upišite novi: ")
        broj = float(input())

    if(broj >= 0.9):
        print("Ocjena pripada kategoriji A!")
    elif(broj >= 0.8):
        print("Ocjena pripada kategoriji B!")
    elif(broj >= 0.7):
        print("Ocjena pripada kategoriji C!")
    elif(broj >= 0.6):
        print("Ocjena pripada kategoriji D!")
    else:
        print("Ocjena pripada kategoriji F!")

except:
    print("An exception occurred...")

