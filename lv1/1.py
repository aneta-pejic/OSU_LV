
def total_euro(satnica, radni_sati):
    return satnica * radni_sati

print("Unesite radne sate:")
radni_sati = float(input())

print("Unesite satnicu:")
satnica = float(input())

print("Radni sati: ",  radni_sati, "h")
print("eura/h: ",  satnica)
print("Ukupno: ",  total_euro(satnica, radni_sati), "eura")
