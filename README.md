
# [SUM](https://sum-project.eu/)

# SimFLEX: Simulation Framework for Feeder Location Evaluation
> This software implements a methodology for assessing the KPIs for on-demand feeder bus services and determining suitable urban areas for their deployment.

## Technology
The software was implemented using Python programming language and its basic libraries
## Description
### The developed method utilizes two main frameworks:
* [ExMAS](https://github.com/RafalKucharskiPK/ExMAS/tree/master/ExMAS)
* [Public Transport queries](https://github.com/RafalKucharskiPK/query_PT)
### The roadmap of the current project for the city:
1) Generate a demand dataset as $Q = {O_i, D_i, T_i}$. Sampling is done according to the travel patterns of the study area (ODM). 
     
 ![OD](https://github.com/anniutina/SimFLEX/blob/main/src/functions.py) 

2) Calculate utilities of different transport modes:
   
  a) Utility of PT trip option from O to D: $U_{PT:O\to \overline{D}}$

  b) Utility of PT trip segment from hub H to D: $U_{PT:H\to \overline{D}}$

  c) Utility of integrated feeder option $F$ (following ExMAS) from O to H:

$$
\begin{aligned}
 U_{F}=U_{PT:H\to \overline{D}} + \underbrace{\beta _{t}\beta _{s}\left ( t _{t}+ \beta _{w}t _{w}\right)}+ASC
\end{aligned}
$$

3) Mode Choice
   * сalculate ASC for the given $E(p_{F})$
   * define the average ASC 
   * recalculate $p_{F}$ for all PT users

4) Demand for F
   * sample travellers choosing F as: $p_{random} < p_{F}$

5) KPIs for F
   * ExMAS-calculated KPIs for F users from O to H
   * system-level KPIs
   * comparative and sensitivity analyzes


## Input:
* [csv file](https://github.com/anniutina/SimFLEX/tree/main/data) with population and address points distribution 
  
* graphml file with [city graph](https://github.com/anniutina/SimFLEX/tree/main/data/graphs)
* the [default](https://github.com/anniutina/SUM/blob/main/data/configs) file with configurations
  
* dbf file with OSM network (available e.g. [here](https://www.interline.io/osm/extracts/))
* zip with GTFS file for the area and date that we query (available e.g. from [gtfs](https://gtfs.ztp.krakow.pl/))
* both OSM and GTFS files should be stored in the data folder

## Output:
* [csv](https://github.com/anniutina/SimFLEX/tree/main/results) with results

## Usage:
* Simulations, KPI estimation and comparative analysis of two SUM areas [notebook](https://github.com/anniutina/SimFLEX/blob/main/simulations.ipynb)
* running the OTP server [notebook](https://github.com/OlhaShulikaUJ/SUM_project/blob/main/PT/run%20OTP%20server-KRK.ipynb)

