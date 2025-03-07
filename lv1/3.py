
brojevi = []

while True:
    unos = input()
    if(unos.lower() == "done"):
        break
    try:
        broj = int(unos)
        brojevi.append(broj)
    except:
        print("Uneseni znak nije broj.")

print("Broj unesenih brojeva: ", len(brojevi))
print("Srednja vrijednost: ", sum(brojevi)/len(brojevi))
print("Minimalna vrijednost: ", min(brojevi))
print("Maximalna vrijednost: ", max(brojevi))
sortiraniBrojevi = sorted(brojevi)
print("Sortirana lista: ", sortiraniBrojevi)