#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
                    Created on Sun Aug 30 22:23:48 2020@author: michael
"""
import errno
import glob
import pandas as pd
import re
import shutil
import string
import time
import tkinter as tk
import tkinter.filedialog as fd
import nltk
import os
import tempfile
from nltk.collocations import AbstractCollocationFinder
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
global range_crit
global freq_crit_list
global filename_input
global wd
global newdir
from collections import Counter
import concurrent.futures

def process_file(filePath, newdir, replacement_dict):
    fileInfo_d = getFileInfo(filePath)
    fileName = os.path.join(newdir, fileInfo_d["fileName"])

    with open(filePath, 'r', encoding="utf8") as file:
        text = file.read().replace("’", "'").lower()
    
    # Tokenize and process text
    tokens = word_tokenize(text)
    processed_text = ' '.join(tokens)

    # Consolidated replacements using regex
    for key, value in replacement_dict.items():
        processed_text = re.sub(r'\b' + re.escape(key) + r'\b', value, processed_text)

    # Write processed text back to file
    with open(filePath, 'w', encoding="utf8") as file:
        file.write(processed_text)


def filterTextLines( text, filter_re_ls ):
    filteredText = ""
    for line in text.splitlines():
        errorCount = 0
        for filter_re in filter_re_ls:
            if re.match( filter_re, line ):
                errorCount += 1
        if errorCount == 0:
            filteredText += line + os.linesep
    return filteredText.strip()

def getFileInfo( relativePath ):
    fullPath = os.path.abspath( relativePath )
    fileName = os.path.basename( relativePath )
    folderPath = re.sub( f"{fileName}$", "", fullPath )
    fileInfo_d = { 
        "fileName": fileName,
        "folderPath": folderPath
        }
    return fileInfo_d

def regexReplaceIt( column, target, replacement ):
    """
    Takes a Pandas column and replaces the target string with the replacement
    """
    return [ re.sub( target, replacement, i) for i in column ]

wd = os.path.dirname(os.path.abspath(__file__))
print(wd)
trigram_measures = nltk.collocations.TrigramAssocMeasures()
m = re.compile("\('[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+'\)")
punctuation = "!()-[]{};:'\"\,<>./?@#$%^&*_~"
print("initial bundle screening")

def get_freq_list():
    global freq_crit_list
    freq_crit_list[0] = int(E1.get())
    freq_crit_list[1] = int(E2.get())
    freq_crit_list[2] = int(E3.get())
    freq_crit_list[3] = int(E4.get())
    freq_crit_list[4] = int(E5.get())
    freq_crit_list[5] = int(E6.get())
    freq_crit_list[6] = int(E7.get())
    print(freq_crit_list)
    return freq_crit_list

def get_range_crit():
    global range_crit
    range_crit = int(E8.get())
    print(range_crit)
    return range_crit

def get_filename():
    global filename_input
    filename_input = str(E9.get())
    print(filename_input)
    return filename_input

def callback():
    global olddir
    dir = fd.askdirectory()
    global corpusfiles
    corpusfiles = os.path.join(dir, '*.txt')
    #print(corpusfiles)
    olddir = os.path.join(dir)
    return olddir, corpusfiles

starttime = time.time()
curr_directory = os.getcwd()

global df_results_cleaned

def functionreport(freqvals):
    freqvals = freqvals.reset_index()
    print(freqvals)
    final_lb_list = list(freqvals['index'])
    print(final_lb_list)
    final_lb_list_df = pd.DataFrame(data = final_lb_list, columns =["Bundle"])
    final_lb_list_df = final_lb_list_df.set_index("Bundle")

    bundledb = pd.read_csv("bundledb.csv", index_col="Bundle", header=0)
    report_merged = final_lb_list_df.merge(bundledb, left_on='Bundle', right_on='Bundle')
    frame4 = final_lb_list_df.join(report_merged, lsuffix='_caller', rsuffix='_other')
    frame4 = frame4.sort_values(by=['Subcat'])
    functional_filename = filename_input + "_functionaltaxonomy_results.csv"
    frame4.to_csv(functional_filename, index='File')

def tokenizecorpus(corp, olddir):
    newdir = os.path.join(olddir, "corpus_copyfix")
    replacement_dict = {
        " 's ": "scontraction ", " 'm ": "mcontraction ", " 're ": "recontraction ",
        " n't ": "ntcontraction ", " 'll ": "llcontraction ", " 'd ": "dcontraction ",
        " 've ": "vecontraction ", " s ": "scontraction ", " m ": "mcontraction ",
        " re ": "recontraction ", " nt ": "ntcontraction ", " ll ": "llcontraction ",
        " d ": "dcontraction ", " ve ": "vecontraction ", "gim me": "gimmecontraction",
        "gon na": "gonnacontraction", "wan na": "wannacontraction",
        "got ta": "gottacontraction", "lem me": "lemmecontraction",
        "wha dd ya": "whaddyacontraction", "wha t cha": "whatchacontraction", "  ": " "
    }

    # Use a ThreadPoolExecutor to parallelize file processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Create a list of tasks for each file
        tasks = [executor.submit(process_file, filePath, newdir, replacement_dict) for filePath in glob.glob(corp)]
        # Wait for all tasks to complete
        for task in concurrent.futures.as_completed(tasks):
            task.result()  # You can handle exceptions here if needed

def interlock_ctrl(listoflbs, freq_threshold, outputcheckname, df_results_cleaned, olddir, newcorpusfiles):
    newdir = os.path.join(olddir, "corpus_copyfix")

    # Process each label in the list
    for lb in listoflbs:
        lb_tokenized = " " + lb + " "

        # First pass: Count occurrences
        for filePath in glob.glob(newcorpusfiles):
            fileInfo_d = getFileInfo(filePath)
            fileName = os.path.join(newdir, fileInfo_d["fileName"])

            with open(filePath, 'r', encoding="utf8") as file:
                content = file.read()
                df_results_cleaned.at[fileName, lb] = content.count(lb_tokenized)

        # Second pass: Replace and write back if above frequency threshold
        if df_results_cleaned[lb].sum() > freq_threshold:
            for filePath in glob.glob(newcorpusfiles):
                fileInfo_d = getFileInfo(filePath)
                fileName = os.path.join(newdir, fileInfo_d["fileName"])

                with open(filePath, 'r', encoding="utf8") as file:
                    content = file.read()

                replace_string = content.replace(lb_tokenized, make_lb_token_traceable(lb_tokenized))

                with open(filePath, 'w', encoding="utf8") as file:
                    file.write(replace_string)

    # Save results to CSV
    current_output_filename = os.path.join(olddir, f"{outputcheckname}check.csv")
    df_results_cleaned.to_csv(current_output_filename, index='File')
    return df_results_cleaned


def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)

def make_lb_token_traceable(input_str):
    # Replace spaces with '+' and add the '<' and '>' at the beginning and end respectively
    return f" <{input_str.replace(' ', '+')}> "

def create_html_files(file_path):
    # Read filenames in the directory and sort them alphabetically
    filenames = sorted([f for f in os.listdir(file_path) if f.endswith('.txt')])

    for filename in filenames:
        # Read the content of each file
        with open(os.path.join(file_path, filename), 'r') as file:
            content = file.read()

        # Function to capitalize the first letter of a sentence
        def capitalize_sentence(sentence):
            return sentence.strip().capitalize()

        # Function to format special strings
        def format_special_string(match):
            text = match.group(0)
            # Removing <, > and replacing + with spaces
            text = text[1:-1].replace('+', ' ')
            # Applying bold and underline
            return f'<b><u>{text}</u></b>'

        # Function to replace contraction patterns
        def replace_contraction(match):
            return f"'{match.group(1)}"

        # Apply transformations
        content = re.sub(r'<[\w+]+>', format_special_string, content)
        content = re.sub(r'([a-z])contraction', replace_contraction, content)

        # Transform content into paragraphs
        paragraphs = [capitalize_sentence(p) + '.' for p in content.split('.') if p]
        formatted_paragraphs = ['<p>' + p + '</p>' for p in paragraphs]

        # Create HTML content
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f0f0f0; }}
                .page-content {{ 
                    background-color: #fff; 
                    margin: 20px; 
                    padding: 20px; 
                    font-size: 18px; 
                }}
                .page-content p {{ 
                    margin-bottom: 15px; 
                }}
                .nav-bar {{ 
                    background-color: #ddd; 
                    width: 200px; 
                    height: 100vh; 
                    overflow-y: scroll; 
                    position: fixed; 
                }}
                .main-content {{ margin-left: 220px; }}
            </style>
        </head>
        <body>
            <div class="nav-bar">
                <ul>
                    {''.join([f'<li><a href="{f[:-4]}.html">{f[:-4]}</a></li>' for f in filenames])}
                </ul>
            </div>
            <div class="main-content">
                <div class="page-content">
                    {''.join(formatted_paragraphs)}
                </div>
            </div>
        </body>
        </html>
        """

        # Write the HTML content to a new file
        with open(os.path.join(file_path, filename[:-4] + '.html'), 'w') as html_file:
            html_file.write(html_content)

def unified_corpus_export(text):
    # Define replacements in a dictionary for efficiency
    replacements = {
        "'": "", "  ": " ", " 's ": "scontraction ", " 'm ": "mcontraction ",
        # Add all other replacements here
        "wha t cha": "whatchacontraction"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.lower()
    return text

def read_ngram_csv(file_path):
    return pd.read_csv(file_path, delimiter=',', names=['Term']).iloc[:, 0].tolist()

def process_ngram_list_text(text, replacement_dict):
    for old, new in replacement_dict.items():
        text = text.replace(old, new)
    return text.lower()

def tokenize_and_update(fileName, text, regex_patterns, ngram_csvs, frame):
    tokens = nltk.Text(word_tokenize(text))
    for bundlelens in range(9, 2, -1):
        pattern = regex_patterns[bundlelens]
        list_grams = ngram_csvs[bundlelens]
        for gram in ngrams(tokens, bundlelens):
            gram_str = str(gram)
            if pattern.match(gram_str) and gram_str in list_grams:
                frame.at[fileName, gram_str] = frame.at[fileName, gram_str] + 1

def lbiap_go():
    replacement_dict = {
        " 's ": "scontraction ", " 'm ": "mcontraction ", " 're ": "recontraction ",
        " n't ": "ntcontraction ", " 'll ": "llcontraction ", " 'd ": "dcontraction ",
        " 've ": "vecontraction ", " s ": "scontraction ", " m ": "mcontraction ",
        " re ": "recontraction ", " nt ": "ntcontraction ", " ll ": "llcontraction ",
        " d ": "dcontraction ", " ve ": "vecontraction ", "gim me": "gimmecontraction",
        "gon na": "gonnacontraction", "wan na": "wannacontraction",
        "got ta": "gottacontraction", "lem me": "lemmecontraction",
        "wha dd ya": "whaddyacontraction", "wha t cha": "whatchacontraction", "  ": " "
    }
    newdir = os.path.join(olddir, "corpus_copyfix")
    copy(olddir, newdir)
    
    global newcorpusfiles
    newcorpusfiles = os.path.join(newdir, "*.txt")
    
    get_freq_list()
    get_range_crit()
    get_filename()
    print(filename_input)    
    global df_results_cleaned
    global testvalue6
    global testvalue7
    tokenizecorpus(newcorpusfiles, olddir)
    outfilename = os.path.join(olddir, "corpus.txt")
    
    # Concatenate files
    with open(outfilename, 'wb') as outfile:
        for filename in glob.glob(newcorpusfiles):
            if filename != outfilename:  # Skip the output file itself
                with open(filename, 'rb') as readfile:
                    shutil.copyfileobj(readfile, outfile)
    
    # Process the concatenated file
    with open(outfilename, 'r', encoding="utf8") as file:
        processed_text = unified_corpus_export(file.read())
    
    tokens_ls_source = word_tokenize(processed_text)
    print(len(tokens_ls_source))
    
    # Remove the concatenated file if no longer needed
    os.remove(outfilename)
    
    print(len(tokens_ls_source))
    
    bundlelens = 9
    threshold = 5
    while bundlelens > 2:
        ninefinder = AbstractCollocationFinder._ngram_freqdist(tokens_ls_source, bundlelens)
        new = pd.DataFrame.from_dict(ninefinder, orient='index')
        new.columns = ['Frequency']
        new.index.name = 'Term'
        if (bundlelens > 5):
            new = (new.where(new['Frequency'] >= freq_crit_list[3])
                       .dropna())
        elif (bundlelens == 5):
            new = (new.where(new['Frequency'] >= freq_crit_list[4])
                      .dropna())
        elif (bundlelens == 4):
            new = (new.where(new['Frequency'] >= freq_crit_list[5])
                      .dropna())
        elif (bundlelens == 3):
            new = (new.where(new['Frequency'] >= freq_crit_list[6])
                      .dropna())
        del new['Frequency']
        lensgrams = str(bundlelens) + "grams.csv"
        lensgramspath = os.path.join("csv", lensgrams)
        new.to_csv(lensgramspath)
        bundlelens = bundlelens - 1
    print("Done")
    files = list(glob.glob(newcorpusfiles))
        
    all_files = glob.glob(os.path.join("csv", "*.csv"))
    
    li = []
    
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    
    files_list = pd.DataFrame(files)
    frame = pd.concat(li, axis=0, ignore_index=False)
    frame = frame.set_index('Term')
    frame = frame.T
    frame['Files'] = files_list.loc[:, 0]
    
    frame = frame.set_index('Files')
    frame = frame.fillna(0)
    
    bundlelens = 9
    ngram_csvs = {i: read_ngram_csv(os.path.join('csv', f'{i}grams.csv')) for i in range(3, 10)}
    regex_patterns = {i: re.compile("\('" + "',\ '".join(["[A-Za-z0-9À-ÖØ-öø-ÿ]+" for _ in range(i)]) + "'\)") for i in range(3, 10)}

    for filePath in glob.glob(newcorpusfiles):
        fileInfo_d = getFileInfo(filePath)
        fileName = os.path.join(newdir, fileInfo_d["fileName"])
        with open(filePath, encoding="utf8") as file:
            text = file.read()

        processed_text = process_ngram_list_text(text, replacement_dict)
        tokenize_and_update(fileName, processed_text, regex_patterns, ngram_csvs, frame)
    
    frame.to_csv("frame2.csv")
    
    print("Frequency Check")
    #freq check subroutine
    results3 = frame.sum()
    results4 = (pd.DataFrame(results3)
                .reset_index())
    results4 = results4.where(results4[0] > 4)
    results4 = results4.dropna()
    results4 = results4.reset_index()
    results5 = list(results4['Term'])
    
    df_final_results = pd.DataFrame()
    for i in results5:
        df_final_results[i] = frame[i]
        
    #range check subroutine
    print("Range Check")
    #Gets zeroes in the column
    testvalue5 = (df_final_results == 0).astype(int).sum(axis=0)
    filecount = len(df_final_results)
    
    #Range test
    columns_to_delete = []
    for index, value in testvalue5.items():
        if int(value) > (filecount - range_crit):
            columns_to_delete.append(index)
    
    df_final_results_exp = df_final_results.copy()
    
    for i in columns_to_delete:
        del df_final_results_exp[i]
    
    #ready for round II
    #print the list of all the column headers 
    #strip punctuation from LB List
    
    for (columnName, columnData) in df_final_results_exp.items(): 
    #    print('Colunm Name : ', columnName)
        no_punc_temp = str(columnName)
        no_punc_temp = no_punc_temp.translate(str.maketrans('', '', string.punctuation))
        df_final_results_exp.rename(columns = {columnName:no_punc_temp}, inplace = True)
    
    new_lb_list = list(df_final_results_exp.columns.values)
    
    length_go = []
    
    for item_go in new_lb_list:
        bundlemeasure = nltk.word_tokenize(item_go)
        length = len(bundlemeasure)
        length_go.append(length)
    
    bundles_by_words = pd.DataFrame(columns=['bundle', 'words'])
    bundles_by_words = bundles_by_words.set_index('bundle')
    bundles_by_words = pd.DataFrame(new_lb_list, columns=['bundle'])
    bundles_by_words['length'] = length_go
    
    #purge results that don't meet frequency criteria
    #9gram check
    print("9gram check")
    
    # Assuming freq_crit_list and df_final_results_exp are defined
    # Initialize dictionary for different n-grams
    n_grams_dict = {}
    columns_to_delete = []
    
    # Loop to filter and convert to lists
    for n in range(9, 2, -1):
        n_grams_dict[n] = bundles_by_words.loc[bundles_by_words['length'] == n, 'bundle'].tolist()
    
    # Loop for frequency check
    for n in range(9, 2, -1):
        print(f"{n}gram check")
        for item in n_grams_dict[n]:
            if df_final_results_exp[item].sum() < freq_crit_list[9-n]:
                columns_to_delete.append(item)
                print(item)
    
    # At this point, columns_to_delete contains all the items to be deleted across all n-grams
    for i in columns_to_delete:
        del df_final_results_exp[i]
    results_initial = filename_input + "_results_initial.csv"
    df_final_results_exp.to_csv(results_initial, index='File')
    
    #move on to reset values
    df_results_reset = df_final_results_exp
    
    for col in df_results_reset.columns:
        df_results_reset[col].values[:] = 0
    
    df_results_cleaned = df_results_reset
    
    #code for finding and replacing each n-gram detected 
    #and then deleting that from the corpus and moving on to each one
    #reprogram it to do an id sweep first and then a deletion sweep
    #add refresh code at each line
    
    # Loop to sort the n-gram lists in the dictionary
    for n in range(9, 2, -1):
        n_grams_dict[n].sort()
    
    tokenizecorpus(newcorpusfiles, olddir)
    # Loop to call interlock_ctrl for each n-gram
    for n in range(9, 2, -1):
        interlock_ctrl(n_grams_dict[n], freq_crit_list[9-n] - 1, f"{n}gram", df_results_cleaned, olddir, newcorpusfiles)

    new_lb_list = list(df_results_cleaned.columns.values)
    new_lb_list.sort()
    
    # Tokenizing items and getting their lengths
    length_go = [len(nltk.word_tokenize(item)) for item in new_lb_list]
    
    # Creating the DataFrame directly with necessary columns
    bundles_by_words3 = pd.DataFrame({
        'bundle': new_lb_list,
        'length': length_go
    })
    
    # Initialize dictionary for n-grams and columns to delete
    n_grams_dict = {}
    columns_to_delete = []
    
    # Filtering and sorting n-gram lists
    for n in range(9, 2, -1):
        n_grams = bundles_by_words3.loc[bundles_by_words3['length'] == n, 'bundle']
        n_grams_list = sorted(n_grams.values.tolist())
        n_grams_dict[n] = n_grams_list
    
        # Frequency check and accumulate columns to delete
        for item in n_grams_list:
            if df_results_cleaned[item].sum() < freq_crit_list[9 - n]:
                columns_to_delete.append(item)
                print(item)
    
    # Writing deleted columns to a file
    # Writing columns to be deleted and deleting them from df_results_cleaned
    deleted_columns_filename = filename_input + "_deleted_columns.txt"
    with open(deleted_columns_filename, 'w') as file:
        for column in sorted(columns_to_delete):
            file.write(column + '\n')
            if column in df_results_cleaned:
                del df_results_cleaned[column]
    
    # Range check subroutine
    print("Range Check")
    testvalue5 = (df_results_cleaned == 0).astype(int).sum(axis=0)
    filecount = len(df_results_cleaned)
    
    # Range test
    columns_to_delete_range = []
    for index, value in testvalue5.items():
        if int(value) > (filecount - range_crit):
            columns_to_delete_range.append(index)
    
    # Copy df_results_cleaned for range check
    df_results_range_check = df_results_cleaned.copy()
    
    # Write to file and delete columns based on range check
    with open(deleted_columns_filename, 'a') as file:  # Open in append mode
        for i in columns_to_delete_range:
            if i in df_results_range_check:
                del df_results_range_check[i]
            file.write(i + "\n")

    replace_dict = {
    'scontraction': "'s", 'mcontraction': "'m", 'recontraction': "'re",
    'ntcontraction': "n't", 'llcontraction': "'ll",
    'dcontraction': "'d", 'vecontraction': "'ve", 'gimmecontraction': "gimme",
    'gonnacontraction': "gonna", 'gottacontraction': "gotta", 'lemmecontraction': "lemme",
    'wannacontraction': "wanna", 'whaddyacontraction': "whaddya", 'whatchacontraction': "whatcha"
    }
    results_final_filename = filename_input + "_results_final.csv"
    print(df_results_range_check)
    df_results_range_check = df_results_range_check.rename(columns=replace_dict)
    testvalue6 = (df_results_range_check > 0).astype(int).sum(axis=0)
    testvalue7 = (df_results_range_check).astype(int).sum(axis=0)
    freqrangereport = testvalue6.to_frame()
    freqrangereport = freqrangereport.reset_index().rename(columns={0:'range'})
    freqvals = testvalue7.to_frame()
    freqvals = freqvals.reset_index().rename(columns={0:'frequency'})
    freqvals['range'] = freqrangereport['range']
    freqvals['words'] = freqvals['index'].str.count(' ') + 1
    freqvals_filename = (filename_input + "_freqvals.csv")

    freqvals['index'] = freqvals['index'].replace(replace_dict, regex=True)

    freqvals = freqvals.set_index('index')
    freqvals = freqvals.rename(columns={'index':'bundle'})
#    freqvals = freqvals.drop(columns=['level_0', 'Index'])
# Replace the strings
    print(freqvals.columns.tolist())
    freqvals.to_csv(freqvals_filename, index='bundle')
    
    #df_results_range_check.to_csv(results_final_filename, index='File')
    df_results_range_check = df_results_range_check.T
    df_results_range_check = df_results_range_check.reset_index()
    print(df_results_range_check.columns.tolist())
    df_results_range_check['index'] = df_results_range_check['index'].replace(replace_dict, regex=True)
    df_results_range_check.to_csv(results_final_filename + "T.csv", index='File')
    df_results_range_check = df_results_range_check.T
    endtime = (time.time() - starttime)
    
    functionreport(freqvals)
    print("Execution time: " + str(endtime))
    print("Done")
    create_html_files(newdir)
    return df_results_range_check, testvalue6, testvalue7

# ---------------------------

window = tk.Tk()
window.title("LBiaP 0.35")
window.rowconfigure(0, weight=1)
window.columnconfigure(1, weight=1)

txt_edit = tk.Text(window)
fr_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
btn_open = tk.Button(fr_buttons, text="Select Corpus", command=callback)
btn_save = tk.Button(fr_buttons, text="Generate LB Data", command=lbiap_go)

btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_save.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

Lfreq = tk.Label(text = "Frequency")
Lfreq.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

L1 = tk.Label(text = "9-grams")
L1.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
L2 = tk.Label(text = "8-grams")
L2.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
L3 = tk.Label(text = "7-grams")
L3.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
L4 = tk.Label(text = "6-grams")
L4.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
L5 = tk.Label(text = "5-grams")
L5.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
L6 = tk.Label(text = "4-grams")
L6.grid(row=7, column=0, sticky="ew", padx=5, pady=5)
L7 = tk.Label(text = "3-grams")
L7.grid(row=8, column=0, sticky="ew", padx=5, pady=5)
L9 = tk.Label(text = " ")
L9.grid(row=9, column=0, sticky="ew", padx=5, pady=5)
L8 = tk.Label(text = "Range")
L8.grid(row=10, column=0, sticky="ew", padx=5, pady=5)
L9 = tk.Label(text = "Filename")
L9.grid(row=11, column=0, sticky="ew", padx=5, pady=5)

freq_crit_list = [5, 5, 5, 5, 10, 20, 40]
range_crit = 5

E1 = tk.Entry(bd = 3)
E1.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
E2 = tk.Entry(bd = 3)
E2.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
E3 = tk.Entry(bd = 3)
E3.grid(row=4, column=1, sticky="ew", padx=5, pady=5)
E4 = tk.Entry(bd = 3)
E4.grid(row=5, column=1, sticky="ew", padx=5, pady=5)
E5 = tk.Entry(bd = 3)
E5.grid(row=6, column=1, sticky="ew", padx=5, pady=5)
E6 = tk.Entry(bd = 3)
E6.grid(row=7, column=1, sticky="ew", padx=5, pady=5)
E7 = tk.Entry(bd = 3)
E7.grid(row=8, column=1, sticky="ew", padx=5, pady=5)
E8 = tk.Entry(textvariable = range_crit)
E8.grid(row=10, column=1, sticky="ew", padx=5, pady=5)
E9 = tk.Entry(bd = 3)
E9.grid(row=11, column=1, sticky="ew", padx=5, pady=5)

E8.insert(0, range_crit)
E1.insert(0, freq_crit_list[0])
E2.insert(0, freq_crit_list[1])
E3.insert(0, freq_crit_list[2])
E4.insert(0, freq_crit_list[3])
E5.insert(0, freq_crit_list[4])
E6.insert(0, freq_crit_list[5])
E7.insert(0, freq_crit_list[6])
E9.insert(0, "filename")

fr_buttons.grid(row=0, column=0, sticky="ns")
window.mainloop()