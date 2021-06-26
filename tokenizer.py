import pandas as pd
import spacy
nlp = spacy.load("en_core_web_md")

def tokenizer(id,fText,l1,l2):
    doc = nlp(fText)
    bdoc = nlp(l1)
    mdoc = nlp(l2)
    ### TOKENIZING ###, ### REMOVING STOP WORDS ### if not token.is_stop
    wList = [token for token in doc]
    bList = [token for token in bdoc]
    mList = [token for token in mdoc]
    ### NORMALIZING ###, ### STEMMING ###, ### LEMMATIZATION ###
    # wList = [token.lemma_ for token in wList]
    ### POS filtering ###
    df = pd.DataFrame(columns=['satzId','Wort','Attribut'], data = {'satzId': id, 'Wort': wList})
    #cast falls nötig von spacy token zu string konvertieren
    df['Wort'] = df['Wort'].astype(str)
    df['Attribut'] = "O"

    #search for all times the first word of the brand occures in the text
    b_brand = df.loc[df['Wort'].str.lower() == str(bList[0]).lower(), 'Attribut']

    #print(b_brand.index.asi8)

    # searching for the Brand Labels

    #if(len(b_brand.index.asi8) == 1):
    #    df.at[b_brand.index.asi8[0], 'Attribut'] = "B-Brand"
    if(len(bList)==1):
        df.loc[df['Wort'].str.lower() == str(bList[0]).lower(), 'Attribut'] = 'B-Brand'
    else:
        for x in range(len(b_brand.index.asi8)):
            #everytime the first Word of the Brand occurs in the Text we need to ckeck the following tokens if it really is the whole Brand Label
            df_to_check = df.iloc[b_brand.index.asi8[x]:b_brand.index.asi8[x]+len(bList)]
            #print(df_to_check)
            # amount of hits have to be the length of the Brand
            hits = 0
            for i, row in enumerate(df_to_check['Wort']):
                if(row.lower() == str(bList[i]).lower()): #hit if the word in the token is the same as in the brand label we search
                    hits += 1
                else:
                    break
            if(hits == len(bList)):
                df.iloc[b_brand.index.asi8[x]:b_brand.index.asi8[x]+len(bList)]['Attribut'] = "I-Brand"
                df.at[b_brand.index.asi8[x], 'Attribut'] = "B-Brand"
                df.at[b_brand.index.asi8[x]+len(bList)-1, 'Attribut'] = "E-Brand"

    #search for all times the first word of the brand occures in the text
    b_modelnumber = df.loc[df['Wort'].str.lower() == str(mList[0]).lower(), 'Attribut']


    if(len(mList) == 1):
        df.loc[df['Wort'].str.lower() == str(mList[0]).lower(), 'Attribut'] = "B-Modelnumber"
    else:
        for x in range(len(b_modelnumber.index.asi8)):
            #everytime the first Word of the Brand occurs in the Text we need to ckeck the following tokens if it really is the whole Brand Label
            df_to_check = df.iloc[b_modelnumber.index.asi8[x]:b_modelnumber.index.asi8[x]+len(mList)]
            #print(df_to_check)
            # amount of hits have to be the length of the Brand
            hits = 0
            for i, row in enumerate(df_to_check['Wort']):
                if(row.lower() == str(mList[i]).lower()): #hit if the word in the token is the same as in the brand label we search
                    hits += 1
                else:
                    break
            if(hits == len(mList)):
                df.iloc[b_modelnumber.index.asi8[x]:b_modelnumber.index.asi8[x]+len(mList)]['Attribut'] = "I-Modelnumber"
                df.at[b_modelnumber.index.asi8[x], 'Attribut'] = "B-Modelnumber"
                df.at[b_modelnumber.index.asi8[x]+len(mList)-1, 'Attribut'] = "E-Modelnumber"


    #for each in bList:
    #    df.loc[df['Wort'] == str(each), 'Attribut'] = "B-Brand"
    return df



text = ("Front controls Easily operate your range stove without reaching over hot Sensi-Technology pots and pans Sensi-Temp Technology Enjoy the same cooking power as a traditional coil with an added safety feature that meets the new UL858 Household Electric Ranges Standard for Safety Lift-up cooktop Quickly clean up spills and remove crumbs from the subtop Two oven racks Feature a durable construction to help accommodate any size or type of cookware Standard clean oven Smooth surface makes cleaning by hand easier One-piece durable handle Strong and long-lasting Chrome drop bowls Contain spills and remove for easy cleaning")
#text2 = ("KitchenAid® 30\" Stainless Look Under Cabinet Hood Liner-UVL5430JSS","LED Lights provides bright, natural-looking light to give you a better view of your food as it cooks. Dishwasher Safe Aluminum Mesh Grease Filter remove grease to keep up with high-heat cooking techniques and feature a durable, easy-to-clean design. Other Feature 3-Speed Push Button Control")

res = (tokenizer(0,text, "Sensi-Temp Technology", "Easily"))

print(res)

print(res.loc[res['Attribut'] == 'B-Brand'])
print(res.loc[res['Attribut'] == 'I-Brand'])
print(res.loc[res['Attribut'] == 'E-Brand'])
print(res.loc[res['Attribut'] == 'B-Modelnumber'])