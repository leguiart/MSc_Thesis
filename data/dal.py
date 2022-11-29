#!/usr/bin/python

import json
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

class Dal:
    def __init__(self):
        # read database configuration
        self.params = config()

    def insert_indicators(self, indicator_mapping, indicator_set):
        insert_list(self.params, "dbo.experimentIndicators", **{k : indicator_mapping[k] for k in indicator_set})
    
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

    