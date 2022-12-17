#!/usr/bin/python

import json
import pickle
import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2.extras import Json
from psycopg2.extensions import register_adapter

from data.config import config

psycopg2.extras.register_uuid()
register_adapter(dict, Json)
register_adapter(np.int64, int)

def insert(conn_params, table_name, **kwargs):
    """ insert a row into a table"""
    sql = """INSERT INTO {table_name}({columns}) VALUES({placeholders}) RETURNING *;"""\
                                                            .format(table_name=table_name, 
                                                            columns = ",".join(kwargs.keys()), 
                                                            placeholders = ','.join(["%s"] * len(kwargs)))
    conn = None
    obj = None
    try:

        # connect to the PostgreSQL database
        conn = psycopg2.connect(**conn_params)
        with conn:
            # create a new cursor
            with conn.cursor() as cur:
                # execute the INSERT statement
                cur.execute(sql, tuple(kwargs.values()))
                # get the inserted row back
                obj = cur.fetchone()
            # commit the changes to the database
            conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        obj = None
    finally:
        if conn is not None:
            # close communication with the database
            conn.close()

    return obj


def insert_list(conn_params, table_name, **kwargs):
    """ insert multiple rows into a table  """
    sql = """INSERT INTO {table_name}({columns})
             VALUES({placeholders});"""\
                .format(table_name=table_name, 
                        columns = ",".join(kwargs.keys()),
                        placeholders = ','.join(["%s"] * len(kwargs)))
    
    first_column = list(kwargs.keys())[0]
    row_count = len(kwargs[first_column])
    row_list = []
    for i in range(row_count):
        row = []
        for column in kwargs.keys():
            row += [kwargs[column][i]]
        row_list += [tuple(row)]

    conn = None
    try:
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**conn_params)
        with conn:
            # create a new cursor
            with conn.cursor() as cur:
                # execute the INSERT statement
                cur.executemany(sql,row_list)
            # commit the changes to the database
            conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            # close communication with the database
            conn.close()

def iter_row(cursor, size=10):
    while True:
        rows = cursor.fetchmany(size)
        if not rows:
            break
        for row in rows:
            yield row

class Dal:
    def __init__(self):
        # read database configuration
        self.params = config()

    def insert_indicators(self, indicator_mapping, indicator_set):
        insert_list(self.params, "dbo.experimentIndicators", **{k : indicator_mapping[k] for k in indicator_set})
    
    def insert_cppns(self, cppns_output_dict):
        insert_list(self.params, "dbo.experimentcppns", **cppns_output_dict)
    
    def insert_indicator_stats(self, stats_dict):
        insert_list(self.params, "dbo.experimentIndicatorStats", **stats_dict)

    def insert_experiment(self, experiment_id, experiment_name, algorithm_id, parameters):
        row = insert(self.params, "dbo.experiments", experiment_id = experiment_id, 
                      experiment_name = experiment_name, algorithm_id = algorithm_id, 
                      parameters = parameters)
        return {"experiment_id" : row[0], "experiment_name" : row[1], "algorithm_id" : row[2], "parameters" : row[3]} 
    
    def insert_experiment_run(self, run_id, run_number, seed, experiment_id):
        row = insert(self.params, "dbo.experimentruns", run_id = run_id, run_number = run_number, seed = seed, experiment_id = experiment_id)
        return {"run_id" : row[0], "run_number" : row[1], "seed" : row[2], "experiment_id" : row[3]} 

    def insert_experiment_archives(self, run_id, generation, f_me_archive, an_me_archive, un_archive, an_archive):
        archives_pickle = {
            "f_me_archive" : f_me_archive,
            "an_me_archive" : an_me_archive,
            "novelty_archive_un" : un_archive,
            "novelty_archive_an" : an_archive
        }
        archives = {
            "f_me_archive" : [],
            "an_me_archive" : [],
            "novelty_archive_un" : [],
            "novelty_archive_an" : []
        }
        # Coverage is the same for all archives
        for i in range(len(f_me_archive)):
            xf = f_me_archive[i]
            # If one is None, all are None
            if xf is not None:
                xf = f_me_archive[i][0]
                xan = an_me_archive[i][0]
                archives["f_me_archive"] += [[xf.md5, xf.id, xf.fitness, xf.unaligned_novelty, xf.aligned_novelty]]
                archives["an_me_archive"] += [[xan.md5, xan.id, xan.fitness, xan.unaligned_novelty, xan.aligned_novelty]]
                # saveToPickle(f"{self.f_me_archive.archive_path}/elite_{i}.pickle", xf)
                # saveToPickle(f"{self.an_me_archive.archive_path}/elite_{i}.pickle", xan)
            else:
                archives["f_me_archive"] += [0]
                archives["an_me_archive"] += [0]

        for xan in an_archive.novelty_archive:
            archives["novelty_archive_an"] += [[xan.md5, xan.id, xan.fitness, xan.unaligned_novelty, xan.aligned_novelty]]
        
        for xun in un_archive.novelty_archive:
            archives["novelty_archive_un"] += [[xun.md5, xun.id, xun.fitness, xun.unaligned_novelty, xun.aligned_novelty]]

        archives_row = self.get_archives([run_id])

        if archives_row["run_id"]:
            self.update_archives(run_id, generation, pickle.dumps(archives_pickle), archives)
        else:
            insert(self.params, "dbo.experimentarchives", run_id = run_id, generation = generation, archives = pickle.dumps(archives_pickle), archives_json = archives)

    def update_archives(self, run_id, generation, archives, archives_json):
        """ update vendor name based on the vendor id """
        sql = """ UPDATE dbo.experimentarchives
                    SET generation = %s,
                    archives = %s,
                    archives_json = %s
                    WHERE run_id = %s"""
        conn = None
        updated_rows = 0
        try:
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**self.params)
            # create a new cursor
            cur = conn.cursor()
            # execute the UPDATE  statement
            cur.execute(sql, (generation, archives, archives_json, run_id))
            # get the number of updated rows
            updated_rows = cur.rowcount
            # Commit the changes to the database
            conn.commit()
            # Close communication with the PostgreSQL database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()

        return updated_rows

    def get_archives(self, run_ids):
        """ query archives row from the dbo.experimentarchives table """
        sql = "SELECT run_id, generation, archives, archives_json FROM dbo.experimentarchives WHERE run_id IN ({placeholders})"\
            .format(placeholders = ','.join(["%s"]*len(run_ids)))
        conn = None
        archives = {"run_id" : [], "generation" : [], "archives" : [],  "archives_json" : []} 
        try:
            conn = psycopg2.connect(**self.params)
            cur = conn.cursor()
            cur.execute(sql, tuple(run_ids))
            rows = cur.fetchall()
            if cur.rowcount > 0:
                for row in rows:
                    archives["run_id"]+=[row[0]]
                    archives["generation"]+=[row[1]] 
                    archives["archives"]+=[pickle.loads(row[2])]  
                    archives["archives_json"]+=[row[3]] 

            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            raise error
        finally:
            if conn is not None:
                conn.close()
        return archives

    def get_archives_json(self, run_ids):
        """ query archives row from the dbo.experimentarchives table """
        sql = "SELECT run_id, generation, archives_json FROM dbo.experimentarchives WHERE run_id IN ({placeholders})"\
            .format(placeholders = ','.join(["%s"]*len(run_ids)))
        conn = None
        archives = {"run_id" : [], "generation" : [], "archives_json" : []} 
        try:
            conn = psycopg2.connect(**self.params)
            cur = conn.cursor()
            cur.execute(sql, tuple(run_ids))
            rows = cur.fetchall()
            if cur.rowcount > 0:
                for row in rows:
                    for column_index, column in enumerate(archives.keys()):
                        archives[column]+=[row[column_index]]

            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            raise error
        finally:
            if conn is not None:
                conn.close()
        return archives


    def get_algorithm(self, name):
        """ query algorithm row from the dbo.algorithms table """
        sql = "SELECT algorithm_id, algorithm_name FROM dbo.algorithms WHERE algorithm_name = %s"
        conn = None
        algorithm = None
        try:
            conn = psycopg2.connect(**self.params)
            cur = conn.cursor()
            cur.execute(sql, (name,))
            row = cur.fetchone()
            if cur.rowcount > 0:
                algorithm = {"algorithm_id" : row[0], "algorithm_name" : row[1]} 

            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return algorithm

    def get_experiment(self, name):
        """ query experiment row from the dbo.experiments table """
        sql = "SELECT experiment_id, experiment_name, algorithm_id, parameters FROM dbo.experiments WHERE experiment_name = %s"
        conn = None
        experiment = None
        try:
            conn = psycopg2.connect(**self.params)
            cur = conn.cursor()
            cur.execute(sql, (name,))
            row = cur.fetchone()
            if cur.rowcount > 0:
                experiment = {"experiment_id" : row[0], "experiment_name" : row[1], "algorithm_id" : row[2], "parameters" : row[3]} 

            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return experiment
    
    def get_experiment_run(self, experiment_id, run_number):
        """ query experiment run row from the dbo.experimentRuns table """
        sql = "SELECT run_id, run_number, seed, experiment_id FROM dbo.experimentruns WHERE experiment_id = %s AND run_number = %s"
        conn = None
        experiment_run = None
        try:
            conn = psycopg2.connect(**self.params)
            cur = conn.cursor()
            cur.execute(sql, (experiment_id, run_number))
            row = cur.fetchone()
            if cur.rowcount > 0:
                experiment_run = {"run_id" : row[0], "run_number" : row[1], "seed" : row[2], "experiment_id" : row[3]} 

            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return experiment_run

    def get_experiment_runs(self, experiment_id):
        """ query experiment rows of a particular experiment from the dbo.experimentRuns table """
        sql = "SELECT run_id, run_number, seed, experiment_id FROM dbo.experimentruns WHERE experiment_id = %s"
        conn = None
        experiment_runs = {"run_id" : [], "run_number" : [], "seed" : [], "experiment_id" : []}
        try:
            conn = psycopg2.connect(**self.params)
            cur = conn.cursor()
            cur.execute(sql, (experiment_id, ))
            rows = cur.fetchall()
            if cur.rowcount > 0:
                for row in rows:
                    for column_index, column in enumerate(experiment_runs.keys()):
                        experiment_runs[column]+=[row[column_index]]

            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            raise error
        finally:
            if conn is not None:
                conn.close()
        return experiment_runs

    def get_experiment_indicator_stats(self, run_id_list, indicator):
        """ query experiment indicator stats of all runs of an experiments from the dbo.experimentindicatorstats table """
        
        sql = """SELECT eis.best, eis.worst, eis.average, eis.std, eis.median, eis.generation, er.run_number 
                 FROM dbo.experimentindicatorstats as eis
                 INNER JOIN dbo.experimentruns as er
                 ON eis.run_id = er.run_id
                 WHERE eis.run_id IN ({placeholders}) AND eis.indicator = %s
                 ORDER BY eis.generation"""\
                    .format(placeholders = ",".join(["%s"]*len(run_id_list)))
        conn = None
        res_dict = {"best":[], "worst":[], "average":[], "std":[], "median":[], "generation":[], "run_number":[]}

        try:
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**self.params)
            cur = conn.cursor()
            cur.execute(sql, tuple(run_id_list + [indicator]))
            for row in iter_row(cur, 1000):
                for column_index, column in enumerate(res_dict.keys()):
                    res_dict[column]+=[row[column_index]]

            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            raise error
        finally:
            if conn is not None:
                conn.close()
        return res_dict

    def get_experiment_stats(self, run_id_list):
        """ query experiment indicator stats of all runs of an experiments from the dbo.experimentindicatorstats table """
        sql = """SELECT eis.indicator, eis.best, eis.worst, eis.average, eis.std, eis.median, eis.generation, er.run_number
                 FROM dbo.experimentindicatorstats as eis
                 INNER JOIN dbo.experimentruns as er
                 ON eis.run_id = er.run_id
                 WHERE eis.run_id IN ({placeholders})
                 ORDER BY eis.generation"""\
                    .format(placeholders = ",".join(["%s"]*len(run_id_list)))
        conn = None
        res_dict = {"indicator":[], "best":[], "worst":[], "average":[], "std":[], "median":[], "generation":[], "run_number":[]}

        try:
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**self.params)
            cur = conn.cursor()
            # for run_id in run_id_list:
            cur.execute(sql, tuple(run_id_list))
            for row in iter_row(cur, 1000):
                for column_index, column in enumerate(res_dict.keys()):
                    res_dict[column]+=[row[column_index]]

            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            raise error
        finally:
            if conn is not None:
                conn.close()
        return res_dict

    def get_experiment_indicator_run_stats(self, run_id, indicator):
        """ query experiment indicator stats of all runs of an experiments from the dbo.experimentindicatorstats table """
        sql = """SELECT eis.best, eis.worst, eis.average, eis.std, eis.median, eis.generation, er.run_number 
                 FROM dbo.experimentindicatorstats as eis
                 INNER JOIN dbo.experimentruns as er
                 ON eis.run_id = er.run_id
                 WHERE eis.run_id = %s AND eis.indicator = %s
                 ORDER BY eis.generation"""
        conn = None
        res_dict = {"best":[], "worst":[], "average":[], "std":[], "median":[], "generation":[], "run_number":[]}

        try:
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**self.params)
            cur = conn.cursor()
            cur.execute(sql, (run_id, indicator))
            for row in iter_row(cur, 1000):
                for column_index, column in enumerate(res_dict.keys()):
                    res_dict[column]+=[row[column_index]]

            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            raise error
        finally:
            if conn is not None:
                conn.close()
        return res_dict

    def get_experiment_run_stats(self, run_id):
        """ query experiment indicator stats of all runs of an experiments from the dbo.experimentindicatorstats table """
        sql = """SELECT eis.indicator, eis.best, eis.worst, eis.average, eis.std, eis.median, eis.generation, er.run_number 
                 FROM dbo.experimentindicatorstats as eis
                 INNER JOIN dbo.experimentruns as er
                 ON eis.run_id = er.run_id
                 WHERE eis.run_id = %s
                 ORDER BY eis.generation"""
        conn = None
        res_dict = {"indicator":[], "best":[], "worst":[], "average":[], "std":[], "median":[], "generation":[], "run_number":[]}

        try:
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**self.params)
            cur = conn.cursor()
            cur.execute(sql, (run_id, ))
            for row in iter_row(cur, 1000):
                for column_index, column in enumerate(res_dict.keys()):
                    res_dict[column]+=[row[column_index]]

            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            raise error
        finally:
            if conn is not None:
                conn.close()
        return res_dict

    def get_experiment_cppns(self, experiment_id_list):
        """ query experiment indicator stats of all runs of an experiments from the dbo.experimentindicatorstats table """
        sql = """SELECT experiment_id, run_id, md5, cppn_outputs
                 FROM dbo.experimentcppns
                 WHERE experiment_id IN ({placeholders})"""\
                    .format(placeholders = ",".join(["%s"]*len(experiment_id_list)))
        conn = None
        res_dict = {"experiment_id":[], "run_id":[], "md5":[], "cppn_output":[]}

        try:
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**self.params)
            cur = conn.cursor()
            # for run_id in run_id_list:
            cur.execute(sql, tuple(experiment_id_list))
            for row in iter_row(cur, 1000):
                for column_index, column in enumerate(res_dict.keys()):
                    if column != "cppn_output":
                        res_dict[column]+=[row[column_index]]
                    else:
                        res_dict[column]+=[pickle.loads(row[column_index])]

            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            raise error
        finally:
            if conn is not None:
                conn.close()
        return res_dict
    