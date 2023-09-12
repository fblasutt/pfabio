# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 08:55:56 2023

Create a tikz picture showing the transitions from one state (no work,
mini job, part-time, full time) to another before-after the reform. 

SB: simulation object before the reform
SP: simulation object after the reform
"""
import numpy as np


def transitions_chart(SB,SP):
    
    #First the main states at baseline:
        
    #(1) no employment
    b1=np.mean(SB['h'][3,:]==0)
    
    #(2) mini-jobs
    b2=np.mean(SB['h'][3,:]==1)
    
    #(3) part-time
    b3=np.mean((SB['h'][3,:]>=2) & (SB['h'][3,:]<=3))
    
    #(4) full time
    b4=np.mean(SB['h'][3,:]==4)
    
    
    #Fill transition matrix between states
    Π=np.zeros((4,4))
    index=np.array([0,1,2,2,3],dtype=np.int32)
    for i in range(5):
        for j in range(5):
            
            i2=index[i]
            j2=index[j]
            
            Π[i2,j2]+=np.mean((SB['h'][3,:]==i) & (SP['h'][3,:]==j))
            
    
    #Create .tex file displyaing the transition between states
    leng=5*Π#width of arrows
    ifl = lambda x, y: x if y>0.0 else r''#do notdraw if no transitions
    
    def p(x): return str('%3.1f' % x)    
    def p2(x): return str('%3.2f' % x)
    chart=r'\begin{tikzpicture}[->,>=stealth,shorten >=1pt,auto,node distance=2.9cm, semithick] '+\
          r'\tikzstyle{state}=[circle,fill=gray,draw=black,text=white]'+\
          r'\node[state, fill=black,     minimum size=52pt, inner sep = -3pt]              (1)               {\small \begin{tabular}{c} No work  \\ {\scriptsize'+p(b1*100)+'}\end{tabular}}; '+\
          r'\node[state, fill=gray,      minimum size=52pt, inner sep = -3pt]              (2) [right of=1]  {\small \begin{tabular}{c} Mini-job \\ {\scriptsize'+p(b2*100)+'}\end{tabular}}; '+\
          r'\node[state, fill=lightgray, text=black, minimum size=52pt, inner sep = -3pt]  (3) [below of=1]  {\small \begin{tabular}{c} Part-time\\ {\scriptsize'+p(b3*100)+'}\end{tabular}}; '+\
          r'\node[state, fill=white,     text=black,  minimum size=52pt, inner sep = -3pt] (4) [right of=3]  {\small \begin{tabular}{c} Full     \\ {\scriptsize'+p(b4*100)+'}\end{tabular}};\path'+\
        ifl(r'(1) edge[line width='+p2(leng[0,0])+'pt,loop left,outer sep=-2pt] node {\scriptsize '   +p(Π[0,0]*100)+'} (1)',Π[0,0])+\
        ifl(r'(1) edge[line width='+p2(leng[0,1])+'pt,bend left=20,outer sep=-2pt] node {\scriptsize '+p(Π[0,1]*100)+'} (2)',Π[0,1])+\
        ifl(r'(1) edge[line width='+p2(leng[0,2])+'pt,outer sep=-2pt] node {\scriptsize '             +p(Π[0,2]*100)+'} (3)',Π[0,2])+\
        ifl(r'(1) edge[line width='+p2(leng[0,3])+'pt,bend left=20,outer sep=-2pt] node {\scriptsize '+p(Π[0,3]*100)+'} (4)',Π[0,3])+\
        ifl(r'(2) edge[line width='+p2(leng[1,0])+'pt,bend right,outer sep=-2pt] node {\scriptsize '   +p(Π[1,0]*100)+'} (1)',Π[1,0])+\
        ifl(r'(2) edge[line width='+p2(leng[1,1])+'pt,loop right=20,outer sep=-2pt] node {\scriptsize '+p(Π[1,1]*100)+'} (2)',Π[1,1])+\
        ifl(r'(2) edge[line width='+p2(leng[1,2])+'pt,outer sep=-2pt] node {\scriptsize '             +p(Π[1,2]*100)+'} (3)',Π[1,2])+\
        ifl(r'(2) edge[line width='+p2(leng[1,3])+'pt,bend left=20,outer sep=-2pt] node {\scriptsize '+p(Π[1,3]*100)+'} (4)',Π[1,3])+\
        ifl(r'(3) edge[line width='+p2(leng[2,0])+'pt,bend left,outer sep=-2pt] node {\scriptsize '   +p(Π[2,0]*100)+'} (1)',Π[2,0])+\
        ifl(r'(3) edge[line width='+p2(leng[2,1])+'pt,bend left=20,outer sep=-2pt] node {\scriptsize '+p(Π[2,1]*100)+'} (2)',Π[2,1])+\
        ifl(r'(3) edge[line width='+p2(leng[2,2])+'pt,loop left=-2pt] node {\scriptsize '             +p(Π[2,2]*100)+'} (3)',Π[2,2])+\
        ifl(r'(3) edge[line width='+p2(leng[2,3])+'pt,bend left=20,outer sep=-2pt] node {\scriptsize '+p(Π[2,3]*100)+'} (4)',Π[2,3])+\
        ifl(r'(4) edge[line width='+p2(leng[3,0])+'pt,bend left,outer sep=-2pt] node {\scriptsize '   +p(Π[3,0]*100)+'} (1)',Π[3,0])+\
        ifl(r'(4) edge[line width='+p2(leng[3,1])+'pt,bend left=20,outer sep=-2pt] node {\scriptsize '+p(Π[3,1]*100)+'} (2)',Π[3,1])+\
        ifl(r'(4) edge[line width='+p2(leng[3,2])+'pt,outer sep=-2pt] node {\scriptsize '             +p(Π[3,2]*100)+'} (3)',Π[3,2])+\
        ifl(r'(4) edge[line width='+p2(leng[3,3])+'pt,loop right=20,outer sep=-2pt] node {\scriptsize '+p(Π[3,3]*100)+'} (4)',Π[3,3])+\
          r'; \end{tikzpicture} '
          
    #Write table to tex file
    with open('C:/Users/Fabio/Dropbox/occupation/model/pfabio/output/transitions_chart.tex', 'w') as f:
        f.write(chart)
        f.close()

    return Π
            
    
    

