from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from collections import Counter
import textdistance
import re

app = Flask(__name__)


words=[]
with open ('autocorrect book.txt','r',encoding='utf-8')as f:
    data=f.read()
    data=data.lower()
    word=re.findall('\w+',data)
    words+=word

v=set(words)
word_freq=Counter(words)
Total_words_freq = sum(word_freq.values())


probs = {}
for k in word_freq.keys():
    probs[k] = word_freq[k] / Total_words_freq


@app.route('/')
def index():
    return render_template('index.html',suggestions=None)

@app.route('/suggest',methods=['POST'])

def suggest():
    keyword = request.form['keyword'].lower()
    if keyword:
        similarities = [1 - textdistance.Jaccard(qval=2).distance(v, keyword) for v in word_freq.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df.columns = ['Word', 'Prob']
        df['Similarity'] = similarities
        suggestions = df.sort_values(['Similarity', 'Prob'], ascending=False)[['Word', 'Similarity']]
        suggestions_list = suggestions.to_dict('records')  # Convert DataFrame to list of dictionaries
        return render_template('index.html', suggestions=suggestions_list)


if __name__ == '__main__':
    app.run(debug=True)

