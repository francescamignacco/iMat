import os, fnmatch, re, io

def sub_word(line): #checks if the line contains one of the wrong expressions and changes it 
    line=re.sub(' Ca. ', ' Ca ', line)
    line=re.sub(' Ca . ', ' Ca ', line)
    line=re.sub(' ca. ', ' ca ', line)
    line=re.sub(' ca . ', ' ca ', line)
    line=re.sub(' et al. ', ' et al ', line)
    line=re.sub(' Et al. ', ' Ca ', line)
    
    return line

def check(lines):
    reff = 0
    text = ""   #final txt that I will write on the file
    intro = 0   #flag that tells me if I have already removed things before intro
    prev = 0    #it counts the number of previous lines which I suspect to be equations
    eq = 0      #tells me if I suspect long equation
    eqline = "" #here I save the line of the equation
    tmp = ""    #string that i will use to replace the current line or/with previous
    for line in lines:
        if (intro==0): #I check if I have reached the Introduction
            if (line.find('Introduction')>-1):
                line=re.sub(r'.*Introduction', 'Introduction\n', line)
                intro=1
        if (intro==1):
            l=len(line)
            if l<10:                #here I do the equation check thing
                prev=prev+1
                tmp=tmp+'\n'+line   #I save everythg here and then set to "" if useless rows
                if(eq==1):
                    eqline=""
                    eq=0            #I was right about previous equation --> I leave in tmp with other stuff
            else:
                if line.find('=')>-1: #for now I remove longer lines only if contain = and in middle
                    if(prev>0):#I assume an eq is preceeded by short stuff (at least a line)
                        eq=1 #I suspect it's an equation and save both separately and together with previous bullshit
                        tmp=tmp+'\n'+line
                        eqline=line
                else:
                    if(prev>2): #I truncate if I find at least 3 rows < 5
                        if(eq==1): #this is one only if the very last line was a suspected equation
                            eq=0
                            tmp=eqline+'\n'+line #this means I was wrong about the equation
                        else:
                            tmp=line
                        prev=0
                    else:
                        tmp=tmp+'\n'+line #for the case prev=1 or 2 (short line but not equation or table)
                    #only if l>5 & I am not a suspected equation
                        prev=0
                    tmp=sub_word(tmp)
                    text=text+' '+tmp
                    tmp=""
        if (line.find('References')>-1 and len(line)<15): #I check if I have reached the ref
            reff = 1
        if (reff==1):
            break
    return text

#da aggiungere : togli tutto dopo references, rimpiazza le parole con una funzione
#togliere tutto dopo References
#togliere le didascalie delle figure (lo vogliamo veramente?)

#read all the filenames in the folder
myfiles=[]
path='/Users/Napster/Documents/iMat/Python/Paper/txt/'
listOfFiles = os.listdir(path)  
pattern = "*.txt"  
for entry in listOfFiles:  
    if fnmatch.fnmatch(entry, pattern):
            myfiles.append(path+entry)

        
#loop on all files 
for fi in myfiles:
    f=open(fi,"r+")
    lines=f.read()
    lines=lines.split('\n')
    testo=check(lines)
    f.close()
    f=open(fi,"w")
    f.write(testo)