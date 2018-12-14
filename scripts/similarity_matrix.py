class SimilarityMatrix:
    """
    A dictionary that holds the similarity matrix.
    Note:
        i represent a user and j represents a movie if the 
        similarity matrix is user based, otherwise if it is
        movie based.
    """

    def __init__(self):
        """
        Initializes an empty similartiy matrix as an empty
        dictionary of dictionaries.
        """
        self.sim_matrix = {}

    def get(self, i, j=None):
        """
        Returns the similarity dictionary for the key to each value if
        value is None, returns the similarity value for the key to
        a specific value if the value is not None.
        Args:
            i: Key
            j: Value
        Returns:
            similarity: The similarity values for the key to each value
                if value is None, the similarity value for the key to
                a the specific value if the value is not None.
        """
        if j is None:
            if i in self.sim_matrix:
                return self.sim_matrix[i]
            return {}
        else:
            if i in self.sim_matrix and j in self.sim_matrix[i]:
                return self.sim_matrix[i][j]
            return 0
        
    def set(self, i, j, sim_value):
        """
        Sets a similarity value in the similarity matrix for a specified key 
        and a specified value.
        Args:
            i: Key
            j: Value
        """
        if not i in self.sim_matrix:
            self.sim_matrix[i] = {}
        self.sim_matrix[i][j] = sim_value
        if not j in self.sim_matrix:
            self.sim_matrix[j] = {}
        self.sim_matrix[j][i] = sim_value

    def contains(self, i, j):
        """
        Checks whether a similarity value in the similarity matrix for a 
        specified key and a specified value exists or not
        Args:
            i: Key
            j: Value
        Returns: 
            True if similarity value exists, False otherwise
        """
        if i in self.sim_matrix and j in self.sim_matrix[i]:
            return True
        return False