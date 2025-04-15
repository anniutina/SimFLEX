import pandas as pd
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import random
import datetime as dt
import requests
from pyproj import Transformer
import geopandas as gpd
import math
from scipy import optimize

from ExMAS import main
from query_PT import main as qpt_main

# To calculate sample size
KRA_17 = 767348     # Krakow population for 2017
KRA_23 = 804200     # Krakow population for 2023
COEF = KRA_23 / KRA_17

def sample_size(od, df_demo, area_pop):
    '''Calculates the number of travellers leaving the area during the rush hour
        as the proportion between zone and area population and their production
        Parameters: od - ODM, df_demo - DataFrame with city population distribution, 
                    area_pop - area population distribution
        Returns: area production
    '''
    a_zns = np.sort(area_pop['zone_NO'].unique()) # area zones
    # zone production, zone population, partial area population
    z_prod, z_pop, p_a_pop = 0, 0, 0  
    p_a_prod = [] # partial area production

    for z_num in a_zns:
        z_prod = od[(od['zone_NO'] == z_num)]['sum'].item()
        z_pop = sum(df_demo[df_demo["zone_NO"] == z_num]["total"])
        p_a_pop = sum(area_pop[area_pop["zone_NO"] == z_num]["total"])
        p_a_prod.append(z_prod * p_a_pop / z_pop)
    return round(sum(p_a_prod) / 2 * COEF, 2)

def reverse_coords(geom):
    '''Reverse the order of lon, lat in a Shapely geometry
        Parameters: geom - Polygon or MultiPolygon object with coords to reverse
        Returns: a new Polygon or MultiPolygon object with reversed coords, lat lon'''
    if type(geom) == Polygon:
        new_exterior = [(y, x) for x, y in geom.exterior.coords]
        new_interior = [[(y, x) for x, y in interior.coords] for interior in geom.interiors]
        return Polygon(new_exterior, new_interior)
    elif type(geom) == MultiPolygon:
        reversed_multi_poly = []
        for poly in geom.geoms:
            reversed_multi_poly.append(reverse_coords(poly))
        return MultiPolygon(reversed_multi_poly)

def transform_coords(source_crs, target_crs, x, y):
    '''transforms point coordinates from the source coordinate system to the target
       default: "EPSG:3857" to "EPSG:4326"
    '''
    return Transformer.from_crs(source_crs, target_crs, always_xy=True).transform(x, y)

def find_containing_polygon(point: Point, gdf: gpd) -> float:
    '''For a given point finds the polygon containing it
    input: point - type Point; gdf - geopandas dataframe
    output: number of polygon that contains point - type int'''
    for i in range(len(gdf)):
        if gdf.geometry[i].contains(point):
            return gdf.NO[i]
    return None

def haversine(loc1, loc2):
    ''' Haversine formula [km]
        coordinates in decimal degrees (y, x), e.g. (19.881557, 50.012738)'''
    # latitude is the y-coordinate, longitude is the x-coordinate
    Earth_radius = 6371  # [km];   R = 3959.87433 [mi]
    lat1, lon1 = np.radians((loc1[0], loc1[1]))
    lat2, lon2 = np.radians((loc2[0], loc2[1]))
    a = np.sin(0.5 * (lat2 - lat1))**2 + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * (lon2 - lon1))**2
    return 2 * Earth_radius * np.arcsin(np.sqrt(a))

def define_demand(sum_areas, df_demo, gdf_centroid, od, od_probs, params, to_csv=False):
    '''Determine travel demand that occurs within the specified city area during the morning rush hour
        input:  sum_areas - SUM areas [geopandas dataframe]
                df_demo - city demographic [csv]
                gdf_centroid - centroids of city zones [geojson]
                od - ODM [excel]
                od_probs - ODM with probabilities [excel]
                to_csv - save results csv format
        output: dictionary with dataframes containing requests ex. {'Skotniki': (O, D, T)} 
    '''
    res = {}
    for i in range(len(sum_areas)):
        if isinstance(sum_areas, pd.Series):
            area = sum_areas
        else:    
            area = sum_areas.loc[i]
        area_pop = df_demo.copy()
        # define, if area polygon contains address points (area population)
        area_pop['inside_poly'] = area_pop.apply(lambda row: 
                                            area.geometry.contains(Point(row['x'], row['y'])), axis=1)
        area_pop = area_pop[area_pop.inside_poly].reset_index(drop=True)
        # repeat rows N times (equal "total"): N rows = N people
        area_pop_repeated = area_pop.loc[area_pop.index.repeat(area_pop.total)]
        # select a sample of origins
        area_sample = area_pop_repeated.sample(round(sample_size(od, df_demo, area_pop))).reset_index(drop=True)
        area_sample.rename(columns = {'x' : 'origin_x', 'y': 'origin_y'}, inplace = True)
        area_sample['probs'] = [None] * len(area_sample)
        area_sample['desti_zones'] = [None] * len(area_sample)
        for n in range(len(area_sample['zone_NO'])):
            area_sample.at[n, 'desti_zones'] = list(od_probs.zone_NO)
            area_sample.at[n, 'probs'] = list(od_probs.loc[area_sample['zone_NO'][n], 1:])
        area_sample['desti_zone'] = area_sample.apply(lambda row: random.choices(row.desti_zones, 
                                                                        weights=row.probs, k=1)[0], axis=1)
        area_sample['destination_x'] = area_sample.apply(lambda row: 
                                    gdf_centroid[gdf_centroid.NO == row.desti_zone].geometry.iloc[0].coords.xy[0][0], axis=1)
        area_sample['destination_y'] = area_sample.apply(lambda row: 
                                    gdf_centroid[gdf_centroid.NO == row.desti_zone].geometry.iloc[0].coords.xy[1][0], axis=1)    
        area_sample['treq'] = pd.NA
        time_format = '%Y-%m-%d %H:%M:%S'
        time_lb = dt.datetime.strptime('2024-03-28 07:45:00', time_format)
        time_ub = dt.datetime.strptime('2024-03-28 08:15:00', time_format)
        area_sample['treq'] = area_sample['treq'].apply(lambda _: time_lb + 
                            dt.timedelta(seconds=np.random.randint(0, (time_ub - time_lb).seconds)))
        requests = area_sample[['origin_x', 'origin_y', 'destination_x', 'destination_y', 'treq']]
        if isinstance(sum_areas, pd.Series):
            res[sum_areas["name"]] = requests
            if to_csv:
                requests.to_csv('requests/reqs_' + str(sum_areas['name']) + '.csv', index=False)
        else:
            res[sum_areas["name"][i]] = requests
            if to_csv:
                requests.to_csv('requests/reqs_' + str(sum_areas['name'][i]) + '.csv', index=False)
    return res

def PT_utility(requests, params):
    if 'walkDistance' in requests.columns:
        requests = requests
        requests['PT_fare'] = 1 + requests.transitTime * params.avg_speed/1000 * params.ticket_price
        requests['u_PT'] = requests['PT_fare'] + \
                           params.VoT * (params.walk_factor * requests.walkDistance / params.speeds.walk +
                                           params.wait_factor * requests.waitingTime +
                                           params.transfer_penalty * requests.transfers + requests.transitTime)
    return requests

def run_OTP(df, OTP_API):
    '''using OpenTripPlanner server returns routes for the given request
    returns dataframe with OTP requests, drops requests for access=False
    ATTENTION to dataframe indexes'''
    df_query = df.copy()
    query = df_query.apply(lambda row: 
                    qpt_main.parse_OTP_response(requests.get(OTP_API, 
                                                params=qpt_main.make_query(row.squeeze())).json()), axis=1)
    names = []
    for q in query:
        if q['success']:
            names = list(q.keys())
            break
        
    if len(names) > 0:
        for name in names:
            vals = []
            for i in range(len(query)):
                if query[i]['success']:
                    vals.append(query[i][name])
                else:
                    vals.append(pd.NA)
            df_query[name] = vals

    # ignore_index=True if indexes must be sorted
    df_query.dropna(inplace=True, ignore_index=True)
    return df_query

def run_ExMAS(df, inData, params, hub=None, degree=8):
    '''input: df - dataframe with columns [origin_x, origin_y, destination_x, destination_y, treq]
              inData - dotMap object with loaded graph and parameters
              hub - tuple with hub coordinates
              degree - params.max_degree - max number of travellers
        output: runs ExMAS, calculates utilities and KPI's
    '''
    params.nP = len(df)  # sample size
    params.max_degree = degree
    inData.requests = df.copy()
    if hub is None:
        inData.requests['destination'] = inData.requests.apply(lambda row: ox.nearest_nodes(inData.G, row['destination_x'], row['destination_y']), axis=1)
    inData.requests['ttrav'] = inData.requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    inData.requests['pax_id'] = list(range(len(inData.requests)))
    
    inData = main.main(inData, params)

def calc_E_p_sum(df, ASC=0):
    df_p = df.copy()
    df_p['u_SUM_OD_'] = df_p.u_SUM_OD + ASC
    df_p['p_SUM_'] = df_p.apply(lambda row: math.exp(-row.u_SUM_OD_) / \
                        (math.exp(-row.u_SUM_OD_) + math.exp(-row.u_PT_OD)), axis=1)
    return df_p.p_SUM_.mean()


def simulate_MSA(gdf_areas, df_demo, gdf_centroid, od, od_probs, hubs, inData, params, OTP_API, 
                 degree=1, N=1, max_iter=100, ASC=2.2, results_period=0) -> dict:
    ''' Perform simulations to evaluate two travel options: - PT and - Feeder bus + PT: 
        - calculate utilities for each traveller of the given area for single and shared rides (ExMAS)
        - evaluate ASC (if ASC = None)
        - apply method of successive averages (MSA) for travel times variable

    input: gdf_areas - SUM areas [geopandas dataframe]
        df_demo - city population distribution [csv]
        gdf_centroid - centroids of city zones [geojson]
        od - ODM [excel], od_probs - dataframe with destination probabilities
        hubs - choseen hub locations [dict]
        inData, params, OTP_API, degree - ExMAS input
        N - number of area sample replications
        max_iter - number of iterations for shared rides
        ASC - alternative specific constant
        results_period - number of iterations after the sistem stabilization for obtaining KPI's (from ExMAS)
    output: dictionary {'area name': {'avg_sim_res': DataFrame}, {'sum_res': DataFrame}, {}, ...}
        avg_sim_res - mean results of N replications of each area samples
            ex. {'Skotniki': {'avg_sim_res': ['tw_PT_OD', 'tw_PT_HD', 'u_PT_OD', 'u_PT_HD', 'u_SUM_OD', 'p_SUM']}}
        sum_res - results of the last iteration for sample of each area:
            ex. {'Skotniki': {'sum_res': [origin_x, origin_y, destination_x, destination_y, treq, u_PT_OD,
                                origin, hub, dist, ttrav, tarr, u_SUM_OD, p_SUM]}}
        kpis_res - DataFrame with KPIs
        avg_times - average expected times [t_expected_i] for all travelers 
        avg_ts_sh, avg_ts_pt - average travel times for shared or PT option
        times - DataFrame with times, ex. {'Skotniki': t_0, ttrav_sh_i, t_expected_i}
        asc_res - asc coefficients for the case when ASC is unknown
        converged_is - list with numbers of iteration, on which the system converged
    '''
    results = {} # main dictionary to store all the results
    
    def iterate(df_sum):
        ''' inner function to perform iterations to evaluate results for options:
            1. before the system stabilizes -> apply MSA
            2. after stabilization -> to obtain KPIs
        '''
        sum_demand = df_sum[df_sum.apply(lambda row: random.random() < row['p_SUM'], axis=1)] # chosen SUM travelers
        print('sum_demand sample', sum_demand.shape[0])
        
        df_sum_sh = sum_demand[['origin_x', 'origin_y', 'destination_x', 'destination_y', 'treq', 'origin', 'destination', 'dist', 't_expected', 'u_PT_OD', 'nTrips']]
        df_sum_sh['nTrips'] = df_sum_sh['nTrips'] + 1

        if sum_demand.shape[0] == 0:
            print(f'no shared trips for ASC {ASC}')
            return df_sum_sh
        
        run_ExMAS(df_sum_sh, inData, params, hub, degree)
        
        # sort data after ExMAS shuffled trips to assign ttrav_sh to SUM travelers
        inData_reqs_sorted = inData.sblts.requests.sort_values('index') 
        inData_reqs_sorted.set_index('index', inplace=True)
        inData_reqs_sorted = inData_reqs_sorted.rename_axis(None)
        df_sum_sh['ttrav_sh'] = inData_reqs_sorted['ttrav_sh']
        df_sum_sh['tarr'] = df_sum_sh.treq + df_sum_sh.apply(
            lambda req: pd.Timedelta(req.ttrav_sh, 's').floor('s'), axis=1)
        df_sum_sh['t_expected'] = (df_sum_sh['nTrips'] - 1) / df_sum_sh['nTrips'] * df_sum_sh.t_expected + \
            df_sum_sh.ttrav_sh / df_sum_sh['nTrips']
    
        # assign indexes of shared trips to column 'indexes_sum' and RESET INDEX to calculate OTP
        df_sum_sh.insert(0, 'indexes_sum', df_sum_sh.index)
        df_sum_sh = df_sum_sh.reset_index(drop=True)

        # Utility for PT HD
        u_pt_hd = df_sum_sh[['indexes_sum', 'origin_x', 'origin_y', 'destination_x', 'destination_y', 'tarr']]
        # change origin x, y to hub coordinates
        u_pt_hd['origin_x'], u_pt_hd['origin_y'] = hub
        # treq for PT_HD = treq + ttrav_sh = tarr
        u_pt_hd['treq'] = pd.to_datetime(u_pt_hd.tarr)
        u_pt_hd = run_OTP(u_pt_hd, OTP_API)
        PT_utility(u_pt_hd, params)

        # drop rows with unsuccessful HD trips only for df_sum_sh (as u_PT_OD is the same as for the single trips)
        df_sum_sh = df_sum_sh.loc[u_pt_hd.index, :] 
        df_sum_sh.reset_index(drop=True, inplace=True)
        u_pt_hd.reset_index(drop=True, inplace=True)
        
        # Utility F OH - ADD t_w from ExMAS.sblts.rides
        df_sum_sh['u_F'] = df_sum_sh.apply(lambda request: request['t_expected'] * params.VoT * params.WtS + \
                                request['dist'] * params.price / 1000, axis=1)
                                
        # df_sum['u_PT_OD'] doesn't change for SUM demand
        df_sum_sh['u_PT_HD'] = u_pt_hd.u_PT
        df_sum_sh['u_SUM_OD'] = df_sum_sh.u_F + u_pt_hd.u_PT + ASC
        df_sum_sh['p_SUM'] = df_sum_sh.apply(lambda row: math.exp(-row.u_SUM_OD) / \
                                (math.exp(-row.u_SUM_OD) + math.exp(-row.u_PT_OD)), axis=1)
        return df_sum_sh

    for _, area in gdf_areas.iterrows():
        key = area["name"]
        dfres = pd.DataFrame()
        df_kpi_res = pd.DataFrame()
        df_avg_times = pd.DataFrame() # to store avg times expected for each replication
        df_avg_ts_sh, df_avg_ts_pt = pd.DataFrame(), pd.DataFrame() # to store avg times for feeder and PT users
        ascs = [] # if ASC=None, store ASC for each sample of each area
        converged_i = [] # the iteration where the system converged
            
        for repl in range(N):
            converged = False
            df_times = pd.DataFrame()
            df_kpis = pd.DataFrame()
            avg_time, avg_t_sh, avg_t_pt = [], [], []
            area_reqs = define_demand(area, df_demo, gdf_centroid, od, od_probs, params)
            df = area_reqs[key].copy() # df with {O, D, Treq} for the area
            hub = hubs[key]
            print(f'replication {repl + 1} for area {key}')

            # OPTION I: Utility for PT OD
            u_pt_od = df.copy()
            u_pt_od = run_OTP(u_pt_od, OTP_API) # define PT routes for each traveller
            PT_utility(u_pt_od, params)  
            
            df = df.loc[u_pt_od.index, :] # select requests with successful OD trips 
            df.reset_index(drop=True, inplace=True)

            # OPTION II: Utility for SUM (SINGLE TRIPS OH + PT HD)
            df_sum = df.copy()

            # Utility for SINGLE TRIPS OH
            df_sum['origin'] = df_sum.apply(lambda row: ox.nearest_nodes(inData.G, row['origin_x'], row['origin_y']), axis=1)
            df_sum['destination'] = ox.nearest_nodes(inData.G, hub[0], hub[1])
            df_sum['dist'] = df_sum.apply(lambda request: inData.skim.loc[request.origin, request.destination], axis=1)
            df_sum['ttrav'] = df_sum['dist'].apply(lambda request: request / params.avg_speed)
            df_sum['tarr'] = df_sum.treq + df_sum.apply(lambda df_sum: pd.Timedelta(df_sum.ttrav, 's').floor('s'), axis=1)
            df_sum['t_expected'] = df_sum['ttrav'] # optimistic travel time

            # Utility for PT HD
            u_pt_hd = df_sum[['origin_x', 'origin_y', 'destination_x', 'destination_y', 'tarr']]
            # change origin x, y to hub coordinates
            u_pt_hd['origin_x'], u_pt_hd['origin_y'] = hub
            # treq for PT_HD = treq + ttrav + transfertime
            u_pt_hd['treq'] = pd.to_datetime(u_pt_hd.tarr) + pd.Timedelta(params.transfertime, unit='s')
            u_pt_hd = run_OTP(u_pt_hd, OTP_API)
            PT_utility(u_pt_hd, params)

            u_pt_od = u_pt_od.loc[u_pt_hd.index, :] # drop rows with unsuccessful HD trips 
            u_pt_od.reset_index(drop=True, inplace=True)
            df_sum = df_sum.loc[u_pt_hd.index, :] 
            df_sum.reset_index(drop=True, inplace=True)
            u_pt_hd.reset_index(drop=True, inplace=True)
            
            df_sum['u_PT_OD'] = u_pt_od.u_PT # utility for PT OD
            df_sum['u_PT_HD'] = u_pt_hd.u_PT # utility for PT HD
            # Utility for SUM OD
            df_sum['u_SUM_OD'] = df_sum.apply(lambda request: request['ttrav'] * params.VoT * params.WtS + \
                                                request['dist'] * params.price / 1000, axis=1) + u_pt_hd.u_PT + ASC if ASC else \
                                 df_sum.apply(lambda request: request['ttrav'] * params.VoT * params.WtS + \
                                                request['dist'] * params.price / 1000, axis=1) + u_pt_hd.u_PT
            # probability of using SUM for each traveler in a single trip
            df_sum['p_SUM'] = df_sum.apply(lambda row: math.exp(-row.u_SUM_OD) / \
                                          (math.exp(-row.u_SUM_OD) + math.exp(-row.u_PT_OD)), axis=1)
            print("p_SUM ",  df_sum.p_SUM.mean())
            
            # ASC calculation
            if not ASC:
                degree = 1
                ascs.append(optimize.fsolve(lambda x: calc_E_p_sum(df_sum, x) - params.expected_prob, 0)[0])
                print(repl, "ASC ", key, ascs[repl])
              
            # times ---------------------------------------------------
            # optimistic ttrav, updated after each ExMAS iteration for travelers using SUM
            df_times['t_0'] = df_sum['ttrav'] # help dataframe for testing
            avg_time.append(df_sum.ttrav.mean()) # optimistic average travel time for ALL travelers before shared travel
                
            df_sum['nTrips'] = 0 # number of shared trips
                
            if degree > 1:
                for i in range(1, max_iter + 1):
                # loop will run until the system will stabilize or until max_iter
                    print('iteration ', i, ' for', key)
                    
                    df_sum_sh = iterate(df_sum)
                    
                    if df_sum_sh.shape[0] == 0:
                        break

                    # times ---------------------------------------------------
                    # test dataframe with times: {t_0, t_ttrav_sh_i, t_expected_i}
                    df_times.loc[df_sum_sh.index, [f'ttrav_sh_{i}']] = df_sum_sh[['ttrav_sh']].values
                    df_times.loc[df_sum_sh.index, [f't_expected_{i}']] = df_sum_sh[['t_expected']].values

                    df_pt = df_sum.copy() # df to store PT travelers
                    df_pt = df_pt.drop(index=df_sum_sh['indexes_sum'])
                    # Update p_SUM and t_expected for trips chosen for SUM in the main table
                    df_sum.loc[df_sum_sh['indexes_sum'], ['p_SUM']] = df_sum_sh[['p_SUM']].values
                    df_sum.loc[df_sum_sh['indexes_sum'], ['t_expected']] = df_sum_sh[['t_expected']].values
                    df_sum.loc[df_sum_sh['indexes_sum'], ['nTrips']] = df_sum_sh[['nTrips']].values
                    avg_time.append(df_sum.t_expected.mean()) # average expected travel times for ALL travelers
                    avg_t_sh.append(df_sum_sh.t_expected.mean())
                    avg_t_pt.append(df_pt.t_expected.mean())
                
                    # MSA condition ---------------------------------------------------
                    if i > 3 and not converged:
                        delta3 = abs(avg_time[i] - avg_time[i - 1]) / avg_time[i - 1]
                        delta2 = abs(avg_time[i - 1] - avg_time[i - 2]) / avg_time[i - 2]
                        delta1 = abs(avg_time[i - 2] - avg_time[i - 3]) / avg_time[i - 3]

                        # check if convergence criterion is met
                        if (delta3 < params.convergence_threshold and
                            delta2 < params.convergence_threshold and
                            delta1 < params.convergence_threshold):
                            converged = True
                            converged_i.append(i)
                            print(f"Convergence reached at iteration {i}")
                            # print("===============================================================")

                # after the system stabilization, obtain KPI's for the results_period
                if converged:
                    # run ExMAS to obtain KPIs
                    for i in range(results_period):
                        print('iteration after converged ', i + 1, ' for', key)
                        
                        df_sum_sh = iterate(df_sum)
                        df_kpis[i] = inData.sblts.res # obtain kpis after the system stabilized
                        print("-----------------------------------------------------")

                        # times ---------------------------------------------------
                        # dataframe with times: {t_0, t_ttrav_sh_i, t_expected_i}
                        df_times.loc[df_sum_sh.index, [f'ttrav_sh_{i}']] = df_sum_sh[['ttrav_sh']].values
                        df_times.loc[df_sum_sh.index, [f't_expected_{i}']] = df_sum_sh[['t_expected']].values

                        # Update p_SUM and t_expected for trips chosen for SUM in the main table
                        df_sum.loc[df_sum_sh['indexes_sum'], ['p_SUM']] = df_sum_sh[['p_SUM']].values
                        df_sum.loc[df_sum_sh['indexes_sum'], ['t_expected']] = df_sum_sh[['t_expected']].values
                        df_sum.loc[df_sum_sh['indexes_sum'], ['nTrips']] = df_sum_sh[['nTrips']].values
                        # avg_time.append(df_sum.t_expected.mean()) # average expected travel times after stabilization
                else:
                    converged_i.append(0)
                    print(f"no convergence after {max_iter} iterations")
                    print("-----------------------------------------------------")

            df_means = pd.DataFrame([[u_pt_od.waitingTime.mean(), u_pt_hd.waitingTime.mean(), u_pt_od.u_PT.mean(),
                                        u_pt_hd.u_PT.mean(), df_sum.u_SUM_OD.mean(), df_sum.p_SUM.mean()]], 
                                        columns=['tw_PT_OD', 'tw_PT_HD', 'u_PT_OD', 'u_PT_HD', 'u_SUM_OD', 'p_SUM'])
            dfres = pd.concat([dfres, df_means], ignore_index=True)
            df_kpi_res = pd.concat([df_kpi_res, df_kpis], axis=1)
            df_avg_times[repl] = avg_time
            df_avg_ts_pt[repl] = avg_t_pt
            df_avg_ts_sh[repl] = avg_t_sh
            print()    
        print("===============================================================")
        
        results[key] = {'avg_sim_res': dfres, 'sum_res': df_sum, 'kpis_res': df_kpi_res, 
                        'avg_times': df_avg_times, 'avg_ts_sh': df_avg_ts_sh, 'avg_ts_pt': df_avg_ts_pt, 
                        'times': df_times, 'asc_res': ascs, 'converged_is': converged_i}
    return results


def calc_KPIs(df):
    ''' calculates such KPI indicaors: 
        del_VehHourTrav - vehicle hours reduction,
        del_PassUtility - travellers utility gains,
        occupancy,
        del_PassHourTrav - passenger hours increase,
        del_fleet_size - potential fleet size reduction
    input: dataframe with KPIs for each area
    output: dataframe with calculated KPI indicators for analysis '''
    for i, row in df.iterrows():
        df.loc[i, 'del_VehHourTrav'] = abs(float(row['VehHourTrav']) - float(row['VehHourTrav_ns'])) / float(row['VehHourTrav_ns'])
        df.loc[i, 'del_PassUtility'] = abs(float(row['PassUtility']) - float(row['PassUtility_ns'])) / float(row['PassUtility_ns'])
        df.loc[i, 'Occupancy'] = float(row['PassHourTrav']) / float(row['VehHourTrav'])
        df.loc[i, 'del_PassHourTrav'] = abs(float(row['PassHourTrav']) - float(row['PassHourTrav_ns'])) / float(row['PassHourTrav_ns'])
        df.loc[i, 'del_fleet_size'] = abs(float(row['fleet_size_shared']) - float(row['fleet_size_nonshared'])) / float(row['fleet_size_nonshared'])
    return df