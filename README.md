Autor: Filip Weigel
Login: xweige01

<h1>PDS - projekt</h1>
Projekt by měl být plně funkční. Obsahuje rozšíření v podobě anomální komunikace. Skript model.py je nastaven na pátý experiment viz dokumentace. Na standartní výstup vypisuje ground truth, dále confusion matrix a přesnost modelu. 
<h2>Potřebné moduly:</h2>
numpy,

mapplotlib,

scipy,

sklearn,

python3

<h2>Použití:</h2>
python3 model.py
python3 generator.py

Skript generator.py generuje do aktuálního adresáře soubor fake_data.csv obsahující náhodně vygenerovanou komunikaci viz dokumentace.
Skript model.py je aktuálně nastaven na pátý experiment opět viz dokumentace. Pro spuštění dalších experimentů je zapotřebí manuální úprava souboru.
Ve složce output jsou připraveny skripty pro spuštění všech experimentů. exp2.py = experiment 2. atd.

<h2>Obsah:</h2>
/dataset/10122018-104Mega.pcapng
/dataset/fake_data.csv
/dataset/input_data.csv
/output/exp1/1.png
/output/exp1/2.png
/output/exp1/3.png
/output/exp2.py
/output/exp2.txt
/output/exp3.py
/output/exp3.txt
/output/exp4.py
/output/exp4.txt
/output/exp5.py
/output/exp5.txt
/output/exp6.py
/output/exp6.txt
/src/generator.py
/src/model.py
/README.md
