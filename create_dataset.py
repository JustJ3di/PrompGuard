import json
import numpy as np




key_check = ["check_copilot_bandit_semgrep","check_gemini_bandit_semgrep","check_deepseekminmax_bandit-semgrep","check_deepseek_bandit_semgrep","check_gpt4_bandit_semgrep","check_gpt3.5_bandit_semgrep"]


#FUNZIONI DI UTILITà
def factorial(a):
    result = 1
    for k in range(2,n+1):
        result *= k
    return result

def binomial(n, k):
    a, b = (k, n-k) if k < n-k else (n-k, k)
    numerator = 1
    for i in range(b+1, n+1):
        numerator *= i
    return numerator / factorial(a)

def metric_pass(n,k,c):
    num = binomial(n-c,k)
    den = binomial(n,k)
    return 1- num/den

with open("DIRTY_DATASET.json","r") as file:
    data = json.load(file)


'''
indice 0 Basic prompt
indice 3 naive
indice 6 CWE-Specific
indice 9 Comprehensive
indice 12 Persona/Memetic 
'''

#PASS@K per ogni prompt per gli indici vedere tabella di sopra.
result =[]
for i in data:
    c = 0 #correct code
    n = 6 #total code
    k = 1 #
    index = 12
    for key in key_check:
        
        if(i[key][index]!=False):
            c += 1
        
    result.append(metric_pass(n,k,c))
    


print("Pass@k "+str(np.mean(result)))


result.clear()

#Vulnerabl@K per ogni prompt per gli indici vedere tabella di sopra.The vulnerable@k metric measures the probability that at least one code snippet out of k generated samples is vulnerable.the model is better if the vulnerable@k score is lower.
result =[]
for i in data:
    
    c = 0 #vulneabiliti found
    n = 6 #total code
    k = 1 #
    index = 13
    for key in key_check:
        if((i[key][index-1]!= False) and (i[key][index]!=0 or i[key][index+1] !=0)):
            c += 1
        
    result.append(metric_pass(n,k,c))

print(np.mean(result))


ordered_data = sorted(data,key=lambda data: data.get('id'))

print(len(ordered_data))


prompt_tecnique = ["Basic","Naive-Secure","CWE-Specific","Comprehensive","Persona/Memetic Proxy"] 


basis_statistic = {}


for i in ordered_data:
    basis_statistic.update({i["id"]:[0,0,0,0,0]})


def mean(a,b):
    return (a+b)/2

for i in ordered_data:
    tmp = []
    for j in key_check:
        
        if(i[j][1]!=-1 and i[j][2] != -1): #Calcolo passe@1k 
            tmp.append(mean(i[j][1],i[j][2])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_basic = np.mean(tmp)
    tmp.clear()

    for j in key_check:
        
        if(i[j][4]!=-1 and i[j][5] != -1):
            tmp.append(mean(i[j][4],i[j][5])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_naive = np.mean(tmp)
    tmp.clear()

    for j in key_check:
        
        if(i[j][7]!=-1 and i[j][8] != -1):
            tmp.append(mean(i[j][7],i[j][8])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_Comp = np.mean(tmp)
    tmp.clear()

    for j in key_check:
        
        if(i[j][10]!=-1 and i[j][11] != -1):
            tmp.append(mean(i[j][10],i[j][11])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_CWE = np.mean(tmp)
    tmp.clear()

    for j in key_check:
        
        if(i[j][13]!=-1 and i[j][14] != -1):
            tmp.append(mean(i[j][13],i[j][14])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_meme = np.mean(tmp)


    for t in tmp:
        basis_statistic.update({i["id"]:{i["Basic"]:med_basic,i["Naive-Secure"]:med_naive,i["CWE-Specific"]:med_CWE,i["Comprehensive"]:med_Comp,i["Persona/Memetic Proxy"]:med_meme}})
  

        


dataset = {}


for i in ordered_data: #PER OGNI CWE, QUINDI ID DEL DATASET
    tmp = []
    for j in key_check:
        
        if(i[j][1]!=-1 and i[j][2] != -1): #risultati basic prompt
            tmp.append(mean(i[j][1],i[j][2])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_basic = np.mean(tmp)
    tmp.clear()

    for j in key_check:
        
        if(i[j][4]!=-1 and i[j][5] != -1): #risultati naive-secure
            tmp.append(mean(i[j][4],i[j][5])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_naive = np.mean(tmp)
    tmp.clear()

    for j in key_check:
        
        if(i[j][7]!=-1 and i[j][8] != -1): #rislultati cwe-specific
            tmp.append(mean(i[j][7],i[j][8])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_Comp = np.mean(tmp)
    tmp.clear()

    for j in key_check:
        
        if(i[j][10]!=-1 and i[j][11] != -1): #risultati chomprenshive
            tmp.append(mean(i[j][10],i[j][11])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_CWE = np.mean(tmp)
    tmp.clear()

    for j in key_check:
       
        if(i[j][13]!=-1 and i[j][14] != -1): #risultati persona/memetic
            tmp.append(mean(i[j][13],i[j][14])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_meme = np.mean(tmp)


    
    for t in tmp:
        dataset.update({i["id"]:{i["Basic"]:med_basic,i["Naive-Secure"]:med_naive,i["CWE-Specific"]:med_CWE,i["Comprehensive"]:med_Comp,i["Persona/Memetic Proxy"]:med_meme}})


new_score_dataset = {}
#bisogna considerare solo i codici sicuri


for i in ordered_data:
    count_secure = 0
    count_code_pass = 0
    for j in key_check:
        
        if (i[j][0] == True):
            count_code_pass+=1
        if(i[j][1]== 0 and i[j][2] ==  0) :#Calcolo passe@1k 
            count_secure += 1  #bisogna inserire un punteggio negativo quando il codice non funziona ?
    
    med_basic = np.float64(count_secure/count_code_pass)

    count_secure = 0
    count_code_pass = 0
    for j in key_check:

        if (i[j][3] == True):
            count_code_pass+=1
        
        if(i[j][4]== 0 and i[j][5] ==  0):
            count_secure += 1 #bisogna inserire un punteggio negativo quando il codice non funziona ?

    med_naive = np.float64(count_secure/count_code_pass)

    count_secure=0
    count_code_pass = 0
    for j in key_check:

        if (i[j][6] == True):
            count_code_pass+=1
        if(i[j][7]== 0 and i[j][8] ==  0):
            count_secure += 1
    med_CWE = np.float64(count_secure/count_code_pass)


    count_secure=0
    count_code_pass = 0
    for j in key_check:
        if (i[j][9] == True):
            count_code_pass+=1        
        if(i[j][10]== 0 and i[j][11] ==  0):
            count_secure +=1
    
    med_Comp = np.float64((count_secure)/count_code_pass)

    count_secure=0
    count_code_pass = 0
    for j in key_check:
        if (i[j][12] == True):
            count_code_pass+=1
        if(i[j][13]== 0 and i[j][14] ==  0):
            count_secure += 1
    med_meme = np.float64((count_secure)/count_code_pass)

   
    new_score_dataset.update({i["id"]:{i["Basic"]:med_basic,i["Naive-Secure"]:med_naive,i["CWE-Specific"]:med_CWE,i["Comprehensive"]:med_Comp,i["Persona/Memetic Proxy"]:med_meme}})
  

#print(new_score_dataset)
all_cwe = []
for i in ordered_data:
    if i["id"].split("_")[0] not in all_cwe:
        all_cwe.append(i["id"].split("_")[0])

print(all_cwe)



with open("Tool/dataset_secure.json","w") as fp:
    json.dump(new_score_dataset,fp = fp)




print("DATASET AGGIORNATO")

print("numero di prompt nel dataset: " + str(len(new_score_dataset)))