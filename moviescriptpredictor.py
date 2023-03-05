import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import random
from sklearn.naive_bayes import BernoulliNB,ComplementNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import tkinter
import math
bestclassifier = ""
sentimator = SentimentIntensityAnalyzer()
frame = tkinter.Tk()
frame.geometry("800x600")
frame.title("Movie Script Predictor")
labelforeground = "#fc0f68"
labelbackground = "#1709fb"
labelfont = ("arial",31,"bold")
buttonforeground = "#725fa9"
buttonbackground = "#52e79c"
buttonfont = ("arial",17,"bold")
menubackground = "#bf4dda"
moviebackground = "#213653"
toplabel = tkinter.Label(frame,foreground = labelforeground,background = labelbackground,font = labelfont)
toplabel.pack()
predictionlabel = tkinter.Label(frame,foreground = labelforeground,background = labelbackground,font = labelfont,width = 200)
predictionlabel.pack()
movies = ["Fargo","Blade Runner","Rush Hour","Star Wars Episode IV","Star Wars Episode V","Star Wars Episode VI","The Big Lebowski"]
filenames = ["F.txt","BR.txt","RH.txt","SW_EpisodeIV.txt","SW_EpisodeV.txt","SW_EpisodeVI.txt","TBL.txt"]
moviebuttons = []
def selectmovie(index):
    frame.configure(background = moviebackground)
    toplabel.configure(text = movies[index])
    for jindex in range(len(movies)):
        moviebuttons[jindex].place(relx = 1.0,rely = 1.0)
    menubutton.place(relx = 0,rely = 0.5)
    phrasebox.pack()
    predictionbutton.pack()
    phil = open(filenames[index],"r")
    isrelevant = False
    characterphrases = {}
    characterfrequencies = {}
    totalphrases = 0
    for line in phil:
        if isrelevant:
            tokens = line.split('"')
            character = tokens[3]
            if character in characterphrases:
                characterphrases[character] += [tokens[5]]
            else:
                characterphrases[character] = [tokens[5]]
            if character in characterfrequencies:
                characterfrequencies[character] += 1
            else:
                characterfrequencies[character] = 1
            totalphrases += 1
        isrelevant = True
    charfreqs = []
    for character in characterfrequencies:
        charfreqs += [[characterfrequencies[character],character]]
    charfreqs.sort()
    phrasesspoken = 0
    phraseratio = 2 / 3
    index = -1
    features = []
    while phrasesspoken < phraseratio * totalphrases:
        character = charfreqs[index][1]
        for phrase in characterphrases[character]:
            features += [(extractfeats(phrase),character)]
        phrasesspoken += charfreqs[index][0]
        index -= 1
    random.shuffle(features)
    trainlen = len(features) >> 1
    classifiers = [BernoulliNB(),ComplementNB(),MultinomialNB(),KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),LogisticRegression(max_iter = 1000),MLPClassifier(max_iter = 1000),AdaBoostClassifier()]
    highestaccuracy = 0
    global bestclassifier
    for sklearnclassifier in classifiers:
        classifier = nltk.classify.SklearnClassifier(sklearnclassifier)
        classifier.train(features[:trainlen])
        accuracy = nltk.classify.accuracy(classifier,features[trainlen:])
        if accuracy > highestaccuracy:
            highestaccuracy = accuracy
            bestclassifier = classifier
for index in range(len(movies)):
    moviebuttons += [tkinter.Button(frame,text = movies[index],foreground = buttonforeground,background = buttonbackground,font = buttonfont,command = lambda index = index:selectmovie(index))]
    moviebuttons[index].pack()
def extractfeats(text):
    compsum = 0
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        compsum += sentimator.polarity_scores(sentence)["compound"]
    numsents = len(sentences)
    if numsents:
        return {"text":text,"numsents":numsents,"numwords":len(nltk.word_tokenize(text)),"avgcomp":compsum / numsents + 1}
    return {"text":text,"numsents":numsents,"numwords":len(nltk.word_tokenize(text)),"avgcomp":1}
def mainmenu():
    frame.configure(background = menubackground)
    toplabel.configure(text = "Select a movie")
    for index in range(len(movies)):
        moviebuttons[index].place(relx = 0.43 + 0.4 * math.cos(index / len(movies) * 2 * math.pi),rely = 0.5 + 0.25 * math.sin(index / len(movies) * 2 * math.pi))
    menubutton.place(relx = 1.0,rely = 1.0)
    phrasebox.pack_forget()
    predictionbutton.pack_forget()
    predictionlabel.pack_forget()
def predict():
    phrase = phrasebox.get()
    predictionlabel.configure(text = 'The character most likely to say "' + phrase + '" is ' + bestclassifier.classify(extractfeats(phrase)),foreground = labelforeground,background = labelbackground,wraplength = 200)
    predictionlabel.pack()
menubutton = tkinter.Button(frame,text = "Back to main menu",foreground = buttonforeground,background = buttonbackground,font = buttonfont,command = mainmenu)
menubutton.pack()
phrasebox = tkinter.Entry(frame)
phrasebox.focus_set()
phrasebox.pack()
predictionbutton = tkinter.Button(frame,text = "Make prediction",background = buttonbackground,foreground = buttonforeground,font = buttonfont,command = predict)
predictionbutton.pack()
mainmenu()
