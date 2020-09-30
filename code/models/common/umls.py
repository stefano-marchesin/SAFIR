# from: https://github.com/p2/ClinicalTrialsNLP


import os

from sqlite import SQLite


class UMLS(object):
    """safety check for UMLS database"""

    @classmethod
    def check_database(cls):
        """check if database is in place and if not, prompts to import it"""
        absolute = os.path.dirname(os.path.realpath(__file__))
        umls_db = os.path.join(absolute, 'databases/umls.db')
        if not os.path.exists(umls_db):
            raise Exception("The UMLS database at {} does not exist. Run the import script `databases/umls.sh'."
                            .format(os.path.abspath(umls_db)))


class UMLSLookup(object):
    """ UMLS lookup functions"""
    sqlite = None
    did_check_dbs = False
    preferred_sources = ['"SNOMEDCT"', '"MTH"', '"MSH"']

    def __init__(self):
        absolute = os.path.dirname(os.path.realpath(__file__))
        self.sqlite = SQLite.get(os.path.join(absolute, 'databases/umls.db'))

    def lookup_code(self, cui, preferred=False):
        """ Return a list with triples that contain:
        - name
        - source
        - semantic type
        by looking it up in our "descriptions" database.
        The "preferred" settings has the effect that only names from SNOMED
        (SNOMEDCD), Metathesaurus (MTH) and MESH (MSH) will be reported. A lookup in
        our "descriptions" table is much faster than combing through the full
        MRCONSO table.
        :returns: A list of triples with (name, sab, sty)
        """
        if cui is None or len(cui) < 1:
            return []
        # lazy UMLS db checking
        if not UMLSLookup.did_check_dbs:
            UMLS.check_database()
            UMLSLookup.did_check_dbs = True
        # take care of negations
        negated = '-' == cui[0]
        if negated:
            cui = cui[1:]
        parts = cui.split('@', 1)
        lookup_cui = parts[0]
        # STR: Name
        # SAB: Abbreviated Source Name
        # STY: Semantic Type
        if preferred:
            sql = 'SELECT STR, SAB, STY FROM descriptions WHERE CUI = ? AND SAB IN ({})'.format(", ".join(UMLSLookup.preferred_sources))
        else:
            sql = 'SELECT STR, SAB, STY FROM descriptions WHERE CUI = ?'
        # return as list
        arr = []
        for res in self.sqlite.execute(sql, (lookup_cui,)):
            if negated:
                arr.append(("[NEGATED] {}".format(res[0], res[1], res[2])))
            else:
                arr.append(res)
        return arr

    def lookup_synonyms(self, cui, preferred=False):
        """return a list with tuples containing:
        - name
        - source
        :returns: A list of tuples with (name, sab)
        """
        if cui is None or len(cui) < 1:
            return []
        # lazy UMLS db checking
        if not UMLSLookup.did_check_dbs:
            UMLS.check_database()
            UMLSLookup.did_check_dbs = True
        # take care of negations
        negated = '-' == cui[0]
        if negated:
            cui = cui[1:]
        parts = cui.split('@', 1)
        lookup_cui = parts[0]
        # STR: Name
        # SAB: Abbreviated Source Name
        if preferred:
            sql = 'SELECT STR, SAB FROM MRCONSO WHERE CUI = ? AND SAB IN ({})'.format(", ".join(UMLSLookup.preferred_sources))
        else:
            sql = 'SELECT STR, SAB FROM MRCONSO WHERE CUI = ?'
        # return a list
        arr = []
        for res in self.sqlite.execute(sql, (lookup_cui,)):
            if negated:
                arr.append(("[NEGATED] {}".format(res[0], res[1], res[2])))
            else:
                arr.append(res)
        return arr

    def lookup_code_meaning(self, cui, preferred=False, no_html=True):
        """ Return a string (an empty string if the cui is null or not found)
        by looking it up in our "descriptions" database.
        The "preferred" settings has the effect that only names from SNOMED
        (SNOMEDCT), Metathesaurus (MTH) and MESH (MSH) will be reported. A lookup in
        our "descriptions" table is much faster than combing through the full
        MRCONSO table.
        """
        names = []
        for res in self.lookup_code(cui, preferred):
            if no_html:
                names.append("{} ({})  [{}]".format(res[0], res[1], res[2]))
            else:
                names.append("{} (<span style=\"color:#090;\">{}</span>: {})".format(res[0], res[1], res[2]))
        comp = ", " if no_html else "<br/>\n"
        return comp.join(names) if len(names) > 0 else ''

    def lookup_code_for_name(self, name, preferred=False):
        """ Tries to find a good concept code for the given concept name.
        Uses our indexed `descriptions` table.
        :returns: A list of triples with (cui, sab, sty)
        """
        if name is None or len(name) < 1:
            return None
        # lazy UMLS db checking
        if not UMLSLookup.did_check_dbs:
            UMLS.check_database()
            UMLSLookup.did_check_dbs = True
        # CUI: Concept-ID
        # STR: Name
        # SAB: Abbreviated Source Name
        # STY: Semantic Type
        if preferred:
            sql = 'SELECT CUI, SAB, STY FROM descriptions WHERE STR LIKE ? AND SAB IN ({})'.format(", ".join(UMLSLookup.preferred_sources))
        else:
            sql = 'SELECT CUI, SAB, STY FROM descriptions WHERE STR LIKE ?'
        # return as list
        arr = []
        for res in self.sqlite.execute(sql, ('%' + name + '%',)):
            arr.append(res)
        return arr

    def restrict_to_ix_concepts(self, ix_concepts, table):
        """create a table restricted to relations between index concepts only"""
        if not UMLSLookup.did_check_dbs:  # lazy UMLS db checking
            UMLS.check_database()
            UMLSLookup.did_check_dbs = True
        # define query to create table containing the subset of concepts' relations restricted to index concepts
        params = ','.join("'{0}'".format(ix_concept) for ix_concept in ix_concepts)
        query = "CREATE TABLE IF NOT EXISTS " + table + " AS " + \
                "SELECT DISTINCT CUI1, CUI2 FROM MRREL WHERE CUI1 IN ({}) AND CUI2 IN ({});".format(params, params)
        # execute query
        self.sqlite.execute(query)
        return True

    def check_table(self, table):
        """check if table exists"""
        if not UMLSLookup.did_check_dbs:  # lazy UMLS db checking
            UMLS.check_database()
            UMLSLookup.did_check_dbs = True
        # define query to check table
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name='{}';".format(table)
        # execute query
        cursor = self.sqlite.execute(query)
        # fetch result from cursor
        res = cursor.fetchone()
        if res:
            return True
        else:
            return False

    def drop_table(self, table):
        """drop table given table's name"""
        if not UMLSLookup.did_check_dbs:  # lazy UMLS db checking
            UMLS.check_database()
            UMLSLookup.did_check_dbs = True
        # define query to drop table
        query = "DROP TABLE IF EXISTS {};".format(table)
        # execute query
        self.sqlite.execute(query)
        return True

    def create_index(self, ix_name, cols, table):
        """create index given column(s) and table names"""
        if not UMLSLookup.did_check_dbs:  # lazy UMLS db checking
            UMLS.check_database()
            UMLSLookup.did_check_dbs = True
        # define query to create index
        query = "CREATE INDEX IF NOT EXISTS {} ON {} ({});".format(ix_name, table, ','.join(cols))
        # execute query
        self.sqlite.execute(query)
        return True

    def compute_num_edges(self, subj, objs, table):
        """compute the number of edges (i.e. connections) between subject concept and list of object concepts"""
        if not UMLSLookup.did_check_dbs:  # lazy UMLS db checking
            UMLS.check_database()
            UMLSLookup.did_check_dbs = True
        # define query to compute number of concept's connections within document
        params = tuple([subj] + objs)
        query = "SELECT COUNT(CUI2) FROM " + table + " WHERE CUI1 = ? AND CUI2 IN (%s);" % ','.join('?' * len(objs))
        # initialize number of edges to 0
        num_edges = 0
        # execute query
        cursor = self.sqlite.execute(query, params)
        # fetch result from cursor
        res = cursor.fetchone()
        if res:
            # update num_edges
            num_edges = res[0]
        return num_edges
