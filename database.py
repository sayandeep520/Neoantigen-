import os
import psycopg2
import psycopg2.extras
import pandas as pd
import streamlit as st


class DatabaseManager:
    """
    Database manager class for handling PostgreSQL database operations.
    This class provides methods for connecting to the database,
    creating tables, and executing queries.
    """
    
    def __init__(self):
        """Initialize the database manager with connection parameters from environment"""
        self.db_url = os.environ.get('DATABASE_URL')
        self.connection = None
        self.cursor = None

    def connect(self):
        """Establish a connection to the database"""
        if self.connection is None or self.connection.closed:
            try:
                self.connection = psycopg2.connect(self.db_url)
                self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
                return True
            except Exception as e:
                st.error(f"Database connection error: {e}")
                return False
        return True

    def close(self):
        """Close the database connection"""
        if self.connection and not self.connection.closed:
            self.cursor.close()
            self.connection.close()

    def execute_query(self, query, params=None, commit=False):
        """
        Execute a SQL query
        
        Args:
            query: SQL query string
            params: Parameters for the query
            commit: Whether to commit the transaction
            
        Returns:
            Query results or None on error
        """
        if not self.connect():
            return None
        
        try:
            self.cursor.execute(query, params or ())
            if commit:
                self.connection.commit()
            return self.cursor
        except Exception as e:
            st.error(f"Query execution error: {e}")
            if commit:
                self.connection.rollback()
            return None

    def fetch_all(self, query, params=None):
        """
        Fetch all rows from a query
        
        Args:
            query: SQL query string
            params: Parameters for the query
            
        Returns:
            List of rows or None on error
        """
        cursor = self.execute_query(query, params)
        if cursor:
            return cursor.fetchall()
        return None

    def fetch_one(self, query, params=None):
        """
        Fetch one row from a query
        
        Args:
            query: SQL query string
            params: Parameters for the query
            
        Returns:
            One row or None on error
        """
        cursor = self.execute_query(query, params)
        if cursor:
            return cursor.fetchone()
        return None

    def fetch_as_dataframe(self, query, params=None):
        """
        Fetch query results as a pandas DataFrame
        
        Args:
            query: SQL query string
            params: Parameters for the query
            
        Returns:
            DataFrame or None on error
        """
        cursor = self.execute_query(query, params)
        if cursor:
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        return None

    def create_tables(self):
        """Create necessary tables if they don't exist"""
        # Create users table
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                email VARCHAR(255) NOT NULL UNIQUE,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """, commit=True)
        
        # Create projects table
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS projects (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                user_id INTEGER REFERENCES users(id),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """, commit=True)
        
        # Create genomic_data table
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS genomic_data (
                id SERIAL PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                data_type VARCHAR(50) NOT NULL, 
                filename VARCHAR(255) NOT NULL,
                description TEXT,
                file_format VARCHAR(50),
                sample_count INTEGER, 
                feature_count INTEGER,
                metadata JSONB,
                file_path VARCHAR(255),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """, commit=True)
        
        # Create analysis_results table
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id SERIAL PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                result_type VARCHAR(50) NOT NULL,
                name VARCHAR(255) NOT NULL, 
                description TEXT,
                parameters JSONB,
                results JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """, commit=True)
        
        # Create model_storage table for saving trained models
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS model_storage (
                id SERIAL PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                model_type VARCHAR(50) NOT NULL,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                parameters JSONB,
                metrics JSONB,
                file_path VARCHAR(255),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """, commit=True)

    def save_genomic_data(self, project_id, data_type, filename, description, file_format, 
                        sample_count, feature_count, metadata, file_path):
        """
        Save genomic data information to the database
        
        Args:
            project_id: Project ID
            data_type: Type of data (genomic, transcriptomic, proteomic)
            filename: Original filename
            description: Data description
            file_format: File format (CSV, TSV, etc.)
            sample_count: Number of samples
            feature_count: Number of features
            metadata: Additional metadata as a dict
            file_path: Path to the stored file
            
        Returns:
            ID of the inserted record or None on error
        """
        query = """
            INSERT INTO genomic_data (
                project_id, data_type, filename, description, file_format, 
                sample_count, feature_count, metadata, file_path
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        params = (
            project_id, data_type, filename, description, file_format,
            sample_count, feature_count, psycopg2.extras.Json(metadata), file_path
        )
        
        cursor = self.execute_query(query, params, commit=True)
        if cursor:
            return cursor.fetchone()[0]
        return None

    def get_genomic_data(self, project_id=None, data_type=None):
        """
        Retrieve genomic data information
        
        Args:
            project_id: Optional project ID to filter by
            data_type: Optional data type to filter by
            
        Returns:
            DataFrame with genomic data information
        """
        query = "SELECT * FROM genomic_data WHERE 1=1"
        params = []
        
        if project_id is not None:
            query += " AND project_id = %s"
            params.append(project_id)
            
        if data_type is not None:
            query += " AND data_type = %s"
            params.append(data_type)
            
        query += " ORDER BY created_at DESC"
        
        return self.fetch_as_dataframe(query, params)

    def save_analysis_result(self, project_id, result_type, name, description, parameters, results):
        """
        Save analysis results to the database
        
        Args:
            project_id: Project ID
            result_type: Type of result (e.g., 'crispr_targets', 'neoantigens')
            name: Result name
            description: Result description
            parameters: Parameters used for the analysis
            results: Analysis results
            
        Returns:
            ID of the inserted record or None on error
        """
        query = """
            INSERT INTO analysis_results (
                project_id, result_type, name, description, parameters, results
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        params = (
            project_id, result_type, name, description,
            psycopg2.extras.Json(parameters), psycopg2.extras.Json(results)
        )
        
        cursor = self.execute_query(query, params, commit=True)
        if cursor:
            return cursor.fetchone()[0]
        return None

    def get_analysis_results(self, project_id=None, result_type=None):
        """
        Retrieve analysis results
        
        Args:
            project_id: Optional project ID to filter by
            result_type: Optional result type to filter by
            
        Returns:
            DataFrame with analysis results
        """
        query = "SELECT * FROM analysis_results WHERE 1=1"
        params = []
        
        if project_id is not None:
            query += " AND project_id = %s"
            params.append(project_id)
            
        if result_type is not None:
            query += " AND result_type = %s"
            params.append(result_type)
            
        query += " ORDER BY created_at DESC"
        
        return self.fetch_as_dataframe(query, params)

    def save_model(self, project_id, model_type, name, description, parameters, metrics, file_path):
        """
        Save model information to the database
        
        Args:
            project_id: Project ID
            model_type: Type of model
            name: Model name
            description: Model description
            parameters: Model parameters
            metrics: Model evaluation metrics
            file_path: Path to the saved model file
            
        Returns:
            ID of the inserted record or None on error
        """
        query = """
            INSERT INTO model_storage (
                project_id, model_type, name, description, parameters, metrics, file_path
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        params = (
            project_id, model_type, name, description,
            psycopg2.extras.Json(parameters), psycopg2.extras.Json(metrics), file_path
        )
        
        cursor = self.execute_query(query, params, commit=True)
        if cursor:
            return cursor.fetchone()[0]
        return None
    
    def create_default_user_and_project(self):
        """Create a default user and project if they don't exist"""
        # Check if default user exists
        user_id = self.fetch_one("SELECT id FROM users WHERE username = %s", ("default_user",))
        
        if not user_id:
            # Create default user
            cursor = self.execute_query(
                "INSERT INTO users (username, email) VALUES (%s, %s) RETURNING id",
                ("default_user", "default@example.com"),
                commit=True
            )
            if cursor:
                user_id = cursor.fetchone()[0]
            else:
                return None
        else:
            user_id = user_id[0]
            
        # Check if default project exists
        project_id = self.fetch_one(
            "SELECT id FROM projects WHERE user_id = %s AND name = %s", 
            (user_id, "Default CRISPR Cancer Immunotherapy Project")
        )
        
        if not project_id:
            # Create default project
            cursor = self.execute_query(
                """
                INSERT INTO projects (name, description, user_id) 
                VALUES (%s, %s, %s) RETURNING id
                """,
                (
                    "Default CRISPR Cancer Immunotherapy Project",
                    "AI-driven CRISPR cancer immunotherapy analysis project",
                    user_id
                ),
                commit=True
            )
            if cursor:
                project_id = cursor.fetchone()[0]
            else:
                return None
        else:
            project_id = project_id[0]
            
        return {
            "user_id": user_id,
            "project_id": project_id
        }


# Create a singleton instance
db_manager = DatabaseManager()
