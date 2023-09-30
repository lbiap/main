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

def tokenizecorpus(corp):
	newdir = os.path.join(olddir, "corpus_copyfix")
	for filePath in glob.glob(corp):
		reader = ""
		fileInfo_d = getFileInfo( filePath )
		fileName = os.path.join(newdir, fileInfo_d["fileName"])
		folderPath = fileInfo_d["folderPath"]
		f = open( filePath, 'r', encoding="utf8")
		reader = ""
		print(fileName)
		reader = f.read()
		reader = reader.replace("’", "'")
		reader = reader.lower()
		reader = word_tokenize(reader)
		reader = ' '.join(str(r) for r in reader)
		reader = reader.replace(" 's ", "scontraction ")
		reader = reader.replace(" 'm ", "mcontraction ")
		reader = reader.replace(" 're ", "recontraction ")
		reader = reader.replace(" n't ", "ntcontraction ")
		reader = reader.replace(" 'll ", "llcontraction ")
		reader = reader.replace(" 'd ", "dcontraction ")
		reader = reader.replace(" 've ", "vecontraction ")
		reader = reader.replace(" s ", "scontraction ")
		reader = reader.replace(" m ", "mcontraction ")
		reader = reader.replace(" re ", "recontraction ")
		reader = reader.replace(" nt ", "ntcontraction ")
		reader = reader.replace(" ll ", "llcontraction ")
		reader = reader.replace(" d ", "dcontraction ")
		reader = reader.replace(" ve ", "vecontraction ")
		reader = reader.replace("gim me", "gimmecontraction")
		reader = reader.replace("gon na", "gonnacontraction")
		reader = reader.replace("wan na", "wannacontraction")
		reader = reader.replace("got ta", "gottacontraction")
		reader = reader.replace("lem me", "lemmecontraction")
		reader = reader.replace("wha dd ya", "whaddyacontraction")
		reader = reader.replace("wha t cha", "whatchacontraction")

		reader = reader.replace("  ", " ")
		f.close()
		f = open( filePath, 'w', encoding="utf8")		
		f.truncate()
		f.write(reader)
		reader = ""
		f.close()


def interlock_ctrl(listoflbs, freq_threshold, outputcheckname, df_results_cleaned):
	newdir = os.path.join(olddir, "corpus_copyfix")
	for lb in listoflbs:
		reader = ""
		f = ""
		replace_string = ""
		lb_tokenized = " "+lb+" "
		for filePath in glob.glob( newcorpusfiles ):
			reader = ""
			fileInfo_d = getFileInfo( filePath )
			fileName = os.path.join(newdir, fileInfo_d["fileName"])
			folderPath = fileInfo_d["folderPath"]
			f = open( filePath, 'r', encoding="utf8")
			reader = ""
			reader = f.read()
			replace_string = ""
			x = reader.count(lb_tokenized)
			df_results_cleaned.at[fileName, lb] = x
			f.close()
			reader = ""
			f = ""
		
#add refresh code to delete only after testing for frequency thresholds
	
		for filePath in glob.glob( newcorpusfiles ):
			reader = ""
			replace_string = ""
			fileInfo_d = getFileInfo( filePath )
			fileName = os.path.join(newdir, fileInfo_d["fileName"])
			folderPath = fileInfo_d["folderPath"]
			f = open( filePath, 'r', encoding="utf8")
			reader = f.read()
			x = reader.count(lb_tokenized)
			if ((df_results_cleaned[lb].sum() > freq_threshold)):
				replace_string = reader.replace(lb_tokenized, ' @ ')
				df_results_cleaned.at[fileName, lb] = x
				f = open( filePath, 'w+', encoding="utf8")
				f.truncate()
				f.write(replace_string)
				reader = ""
				f.close()
			f = ""
			replace_string = ""
	current_output_filename = str(outputcheckname)+"check.csv"
	df_results_cleaned.to_csv( current_output_filename, index='File')
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


def lbiap_go():
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
	tokenizecorpus(newcorpusfiles)
	outfilename = os.path.join(olddir, "corpus.txt")
	with open(outfilename, 'wb') as outfile:
	    for filename in glob.glob(newcorpusfiles):
	        if filename == outfilename:
	            # don't want to copy the output into the output
	            continue
	        with open(filename, 'rb') as readfile:
	            shutil.copyfileobj(readfile, outfile)
	
	text = open( outfilename, encoding="utf8" ).read()
	text = filterTextLines( text, [ "^fileName:", "^folderPath:" ] )
	text = text.replace("'", "")
	text = text.replace("  ", " ")
	text = text.replace(" 's ", "scontraction ")
	text = text.replace(" 'm ", "mcontraction ")
	text = text.replace(" 're ", "recontraction ")
	text = text.replace(" n't ", "ntcontraction ")
	text = text.replace(" 'll ", "llcontraction ")
	text = text.replace(" 'd ", "dcontraction ")
	text = text.replace(" 've ", "vecontraction ")
	text = text.replace(" s ", "scontraction ")
	text = text.replace(" m ", "mcontraction ")
	text = text.replace(" re ", "recontraction ")
	text = text.replace(" nt ", "ntcontraction ")
	text = text.replace(" ll ", "llcontraction ")
	text = text.replace(" d ", "dcontraction ")
	text = text.replace(" ve ", "vecontraction ")
	text = text.replace("gim me", "gimmecontraction")
	text = text.replace("gon na", "gonnacontraction")
	text = text.replace("wan na", "wannacontraction")
	text = text.replace("got ta", "gottacontraction")
	text = text.replace("lem me", "lemmecontraction")
	text = text.replace("wha dd ya", "whaddyacontraction")
	text = text.replace("wha t cha", "whatchacontraction")
	text = text.lower()
	tokens_ls_source = word_tokenize(text)
	#print((tokens_ls_source))
	os.remove(os.path.join(olddir, "corpus.txt"))
	
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
	for filePath in glob.glob(newcorpusfiles):
		fileInfo_d = getFileInfo( filePath )
		fileName = os.path.join(newdir, fileInfo_d["fileName"])
		print(fileName)
		folderPath = fileInfo_d["folderPath"]
		text = open( filePath, encoding="utf8" ).read()
		text = filterTextLines( text, [ "^fileName:", "^folderPath:" ] )
		text = text.replace("'", "")
		text = text.replace("  ", " ")
		text = text.replace(" 's ", "scontraction ")
		text = text.replace(" 'm ", "mcontraction ")
		text = text.replace(" 're ", "recontraction ")
		text = text.replace(" n't ", "ntcontraction ")
		text = text.replace(" 'll ", "llcontraction ")
		text = text.replace(" 'd ", "dcontraction ")
		text = text.replace(" 've ", "vecontraction ")
		text = text.replace(" s ", "scontraction ")
		text = text.replace(" m ", "mcontraction ")
		text = text.replace(" re ", "recontraction ")
		text = text.replace(" nt ", "ntcontraction ")
		text = text.replace(" ll ", "llcontraction ")
		text = text.replace(" d ", "dcontraction ")
		text = text.replace(" ve ", "vecontraction ")
		text = text.replace("gim me", "gimmecontraction")
		text = text.replace("gon na", "gonnacontraction")
		text = text.replace("wan na", "wannacontraction")
		text = text.replace("got ta", "gottacontraction")
		text = text.replace("lem me", "lemmecontraction")
		text = text.replace("wha dd ya", "whaddyacontraction")
		text = text.replace("wha t cha", "whatchacontraction")
		text = text.lower()
		tokens_ls = word_tokenize(text)
		tokens_ls = nltk.Text(tokens_ls)
		bundlelens = 9
		current_file = fileName
		while bundlelens > 2:
			if bundlelens == 9:
				m = re.compile("\('[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+'\)")
				fpath = os.path.join('csv', '9grams.csv')
				xgrams = pd.read_csv(fpath, delimiter=',', names=['Term'])
				list_grams_3 = xgrams['Term'].tolist()
			elif bundlelens == 8:
				m = re.compile("\('[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+'\)")
				fpath = os.path.join('csv', '8grams.csv')
				xgrams = pd.read_csv(fpath, delimiter=',', names=['Term'])
				list_grams_3 = xgrams['Term'].tolist()
			elif bundlelens == 7:
				m = re.compile("\('[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+'\)")
				fpath = os.path.join('csv', '7grams.csv')
				xgrams = pd.read_csv(fpath, delimiter=',', names=['Term'])
				list_grams_3 = xgrams['Term'].tolist()
			elif bundlelens == 6:
				m = re.compile("\('[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+'\)")
				fpath = os.path.join('csv', '6grams.csv')
				xgrams = pd.read_csv(fpath, delimiter=',', names=['Term'])
				list_grams_3 = xgrams['Term'].tolist()
			elif bundlelens == 5:
				m = re.compile("\('[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+'\)")
				fpath = os.path.join('csv', '5grams.csv')
				xgrams = pd.read_csv(fpath, delimiter=',', names=['Term'])
				list_grams_3 = xgrams['Term'].tolist()
			elif bundlelens == 4:
				m = re.compile("\('[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+'\)")
				fpath = os.path.join('csv', '4grams.csv')
				xgrams = pd.read_csv(fpath, delimiter=',', names=['Term'])
				list_grams_3 = xgrams['Term'].tolist()
			elif bundlelens == 3:
				m = re.compile("\('[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+',\ '[A-Za-z0-9À-ÖØ-öø-ÿ]+'\)")
				fpath = os.path.join('csv', '3grams.csv')
				xgrams = pd.read_csv(fpath, delimiter=',', names=['Term'])
				list_grams_3 = xgrams['Term'].tolist()
			
			tetragrams = ngrams(tokens_ls, bundlelens)
			tetra_dict = list(tetragrams)
			for i in tetra_dict:
				j = str(i)
	
				if (m.match(j) != None):
	#				print("fun begins")
					if (j in list_grams_3):
						frame.at[fileName, j] = (frame.at[fileName, j] + 1)
						#print(j)
	
			bundlelens = bundlelens - 1
			xgrams = pd.DataFrame()
	
	
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
	
	# print the list of all the column headers 
	
	#strip punctuation from LB List
	
	for (columnName, columnData) in df_final_results_exp.items(): 
	#	print('Colunm Name : ', columnName)
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
	
	
	nine_grams_neo = bundles_by_words.loc[bundles_by_words['length'] == 9]
	eight_grams_neo = bundles_by_words.loc[bundles_by_words['length'] == 8]
	seven_grams_neo = bundles_by_words.loc[bundles_by_words['length'] == 7]
	six_grams_neo = bundles_by_words.loc[bundles_by_words['length'] == 6]
	five_grams_neo = bundles_by_words.loc[bundles_by_words['length'] == 5]
	four_grams_neo = bundles_by_words.loc[bundles_by_words['length'] == 4]
	three_grams_neo = bundles_by_words.loc[bundles_by_words['length'] == 3]
	
	new_9grams_list = nine_grams_neo['bundle'].values.tolist()
	new_8grams_list = eight_grams_neo['bundle'].values.tolist()
	new_7grams_list = seven_grams_neo['bundle'].values.tolist()
	new_6grams_list = six_grams_neo['bundle'].values.tolist()
	new_5grams_list = five_grams_neo['bundle'].values.tolist()
	new_4grams_list = four_grams_neo['bundle'].values.tolist()
	new_3grams_list = three_grams_neo['bundle'].values.tolist()
	
	#purge results that don't meet frequency criteria
	
	#9gram check
	print("9gram check")
	
	columns_to_delete = []
	for item in new_9grams_list:
		if ((df_final_results_exp[item].sum() < freq_crit_list[0])):
			columns_to_delete.append(item)
			print(item)
	
	for item in new_8grams_list:
		if ((df_final_results_exp[item].sum() < freq_crit_list[1])):
			columns_to_delete.append(item)
			print(item)
	
	for item in new_7grams_list:
		if ((df_final_results_exp[item].sum() < freq_crit_list[2])):
			columns_to_delete.append(item)
			print(item)
	
	for item in new_6grams_list:
		if ((df_final_results_exp[item].sum() < freq_crit_list[3])):
			columns_to_delete.append(item)
			print(item)
	
	for item in new_5grams_list:
		if ((df_final_results_exp[item].sum() < freq_crit_list[4])):
			columns_to_delete.append(item)
			print(item)
	
	for item in new_4grams_list:
		if ((df_final_results_exp[item].sum() < freq_crit_list[5])):
			columns_to_delete.append(item)
			print(item)
	
	for item in new_3grams_list:
		if ((df_final_results_exp[item].sum() < freq_crit_list[6])):
			columns_to_delete.append(item)
			print(item)
	
	
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
	
	new_9grams_list.sort()
	new_8grams_list.sort()
	new_7grams_list.sort()
	new_6grams_list.sort()
	new_5grams_list.sort()
	new_4grams_list.sort()
	new_3grams_list.sort()
	
	tokenizecorpus(newcorpusfiles)
	interlock_ctrl(new_9grams_list, (freq_crit_list[0] - 1), "9gram", df_results_cleaned)
	interlock_ctrl(new_8grams_list, (freq_crit_list[1] - 1), "8gram", df_results_cleaned)
	interlock_ctrl(new_7grams_list, (freq_crit_list[2] - 1), "7gram", df_results_cleaned)
	interlock_ctrl(new_6grams_list, (freq_crit_list[3] - 1), "6gram", df_results_cleaned)
	interlock_ctrl(new_5grams_list, (freq_crit_list[4] - 1), "5gram", df_results_cleaned)
	interlock_ctrl(new_4grams_list, (freq_crit_list[5] - 1), "4gram", df_results_cleaned)
	interlock_ctrl(new_3grams_list, (freq_crit_list[6] - 1), "3gram", df_results_cleaned)
	
	
	new_lb_list = list(df_results_cleaned.columns.values)
	new_lb_list.sort()
	
	length_go = []
	
	for item_go in new_lb_list:
		bundlemeasure = nltk.word_tokenize(item_go)
		length = len(bundlemeasure)
		length_go.append(length)
	
	
	bundles_by_words3 = pd.DataFrame(columns=['bundle', 'words'])
	bundles_by_words3 = bundles_by_words3.set_index('bundle')
	bundles_by_words3 = pd.DataFrame(new_lb_list, columns=['bundle'])
	bundles_by_words3['length'] = length_go
	
	
	nine_grams_neo = bundles_by_words3.loc[bundles_by_words3['length'] == 9]
	eight_grams_neo = bundles_by_words3.loc[bundles_by_words3['length'] == 8]
	seven_grams_neo = bundles_by_words3.loc[bundles_by_words3['length'] == 7]
	six_grams_neo = bundles_by_words3.loc[bundles_by_words3['length'] == 6]
	five_grams_neo = bundles_by_words3.loc[bundles_by_words3['length'] == 5]
	four_grams_neo = bundles_by_words3.loc[bundles_by_words3['length'] == 4]
	three_grams_neo = bundles_by_words3.loc[bundles_by_words3['length'] == 3]
	
	new_9grams_list = nine_grams_neo['bundle'].values.tolist()
	new_8grams_list = eight_grams_neo['bundle'].values.tolist()
	new_7grams_list = seven_grams_neo['bundle'].values.tolist()
	new_6grams_list = six_grams_neo['bundle'].values.tolist()
	new_5grams_list = five_grams_neo['bundle'].values.tolist()
	new_4grams_list = four_grams_neo['bundle'].values.tolist()
	new_3grams_list = three_grams_neo['bundle'].values.tolist()
	
	new_9grams_list.sort()
	new_8grams_list.sort()
	new_7grams_list.sort()
	new_6grams_list.sort()
	new_5grams_list.sort()
	new_4grams_list.sort()
	new_3grams_list.sort()
	
	
	#purge results that don't meet frequency criteria
	
	#9gram check
	
	columns_to_delete = []
	for item in new_9grams_list:
		if ((df_results_cleaned[item].sum() < freq_crit_list[0])):
			columns_to_delete.append(item)
			print(item)
	
	for item in new_8grams_list:
		if ((df_results_cleaned[item].sum() < freq_crit_list[1])):
			columns_to_delete.append(item)
			print(item)
	
	for item in new_7grams_list:
		if ((df_results_cleaned[item].sum() < freq_crit_list[2])):
			columns_to_delete.append(item)
			print(item)
	
	for item in new_6grams_list:
		if ((df_results_cleaned[item].sum() < freq_crit_list[3])):
			columns_to_delete.append(item)
			print(item)
	
	for item in new_5grams_list:
		if ((df_results_cleaned[item].sum() < freq_crit_list[4])):
			columns_to_delete.append(item)
			print(item)
	
	for item in new_4grams_list:
		if ((df_results_cleaned[item].sum() < freq_crit_list[5])):
			columns_to_delete.append(item)
			print(item)
	
	for item in new_3grams_list:
		if ((df_results_cleaned[item].sum() < freq_crit_list[6])):
			columns_to_delete.append(item)
			print(item)
	deleted_columns_filename = filename_input + "_deleted_columns.txt"
	MyFile=open(deleted_columns_filename,'w')
	
	columns_to_delete.sort()
	
	for i in columns_to_delete:
		del df_results_cleaned[i]
		MyFile.write(i+"\n")
	 
	
	#range check subroutine
	print("Range Check")
	#Gets zeroes in the column
	testvalue5 = (df_results_cleaned == 0).astype(int).sum(axis=0)
	filecount = len(df_results_cleaned)
	
	#Range test
	columns_to_delete = []
	for index, value in testvalue5.items():
		if int(value) > (filecount - range_crit):
			columns_to_delete.append(index)
	global df_results_range_check
	df_results_range_check = df_results_cleaned.copy()
	
	for i in columns_to_delete:
		del df_results_range_check[i]
		MyFile.write(i+"\n")
	
	MyFile.close()
	replace_dict = {
    'scontraction': "'s",
    'mcontraction': "'m",
    'recontraction': "'re",
    'ntcontraction': "n't",
    'llcontraction': "'ll",
    'dcontraction': "'d",
    'vecontraction': "'ve",
    'gimmecontraction': "gimme",
    'gonnacontraction': "gonna",
    'gottacontraction': "gotta",
    'lemmecontraction': "lemme",
    'wannacontraction': "wanna",
    'whaddyacontraction': "whaddya",
    'whatchacontraction': "whatcha"
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
#	freqvals = freqvals.drop(columns=['level_0', 'Index'])


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
	return df_results_range_check, testvalue6, testvalue7

# ---------------------------





window = tk.Tk()
window.title("LBiaP 0.26")
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
