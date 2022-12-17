#!/usr/bin/python

import psycopg2
from config import config


def create_tables():
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE SCHEMA IF NOT EXISTS dbo
            AUTHORIZATION master;
        """,

        """
        CREATE TABLE IF NOT EXISTS dbo.algorithms (
            algorithm_id UUID UNIQUE NOT NULL PRIMARY KEY,
            algorithm_name VARCHAR(50) UNIQUE NOT NULL,
            description TEXT NULL
        );""",

        """
        CREATE TABLE IF NOT EXISTS dbo.experiments (
            experiment_id UUID UNIQUE NOT NULL PRIMARY KEY,
            experiment_name VARCHAR(50) UNIQUE NOT NULL,
            algorithm_id UUID NOT NULL,
            parameters JSON NOT NULL,
            description TEXT NULL,
            FOREIGN KEY (algorithm_id)
                REFERENCES dbo.algorithms (algorithm_id)
        );
        """,

        """
        CREATE TABLE IF NOT EXISTS dbo.experimentRuns(
            run_id UUID UNIQUE NOT NULL PRIMARY KEY,
            run_number INT NOT NULL,
            seed BIGINT NOT NULL,
            experiment_id UUID NOT NULL,
            FOREIGN KEY (experiment_id)
                REFERENCES dbo.experiments (experiment_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dbo.experimentArchives(
            run_id UUID UNIQUE NOT NULL PRIMARY KEY,
            generation INT,
            archives BYTEA NOT NULL,
            archives_json JSON NULL,
            FOREIGN KEY (run_id)
                REFERENCES dbo.experimentRuns(run_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dbo.experimentIndicators(
            individual_id INT NOT NULL,
            md5 VARCHAR(32) NOT NULL,
            generation INT NOT NULL,
            population_type VARCHAR(20) NULL,
            run_id UUID NOT NULL,
            fitness NUMERIC,
            unaligned_novelty NUMERIC,
            aligned_novelty NUMERIC,
            gene_diversity NUMERIC,
            control_gene_div NUMERIC,
            morpho_gene_div NUMERIC,
            morpho_div NUMERIC,
            endpoint_div NUMERIC,
            trayectory_div NUMERIC,
            inipoint_x NUMERIC,
            inipoint_y NUMERIC,
            inipoint_z NUMERIC,
            endpoint_x NUMERIC,
            endpoint_y NUMERIC,
            endpoint_z NUMERIC,
            trayectory_x NUMERIC,
            trayectory_y NUMERIC,
            trayectory_z NUMERIC,
            morphology_active NUMERIC,
            morphology_passive NUMERIC,
            control_cppn_nodes NUMERIC,
            control_cppn_edges NUMERIC,
            control_cppn_ws NUMERIC,
            morpho_cppn_nodes NUMERIC,
            morpho_cppn_edges NUMERIC,
            morpho_cppn_ws NUMERIC,
            simplified_gene_div NUMERIC,
            simplified_gene_ne_div NUMERIC,
            simplified_gene_nws_div NUMERIC,
            FOREIGN KEY (run_id)
                REFERENCES dbo.experimentRuns(run_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dbo.experimentIndicatorStats(
            indicator VARCHAR(50) NOT NULL, 
            best NUMERIC, 
            worst NUMERIC, 
            average NUMERIC, 
            std NUMERIC, 
            median NUMERIC, 
            generation INT NOT NULL,
            run_id UUID NOT NULL,
            FOREIGN KEY (run_id)
                REFERENCES dbo.experimentRuns(run_id)
        );
        """,
        """
        INSERT INTO dbo.algorithms (algorithm_id, algorithm_name)
        VALUES
            (gen_random_uuid(), 'SO'),
            (gen_random_uuid(), 'QN-MOEA'),
            (gen_random_uuid(), 'MAP-ELITES'),
            (gen_random_uuid(), 'NSLC'),
            (gen_random_uuid(), 'MNSLC')
        ON CONFLICT (algorithm_name) DO UPDATE SET algorithm_name = EXCLUDED.algorithm_name;
        """)
    conn = None
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    create_tables()