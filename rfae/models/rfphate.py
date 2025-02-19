from .rfgap import RFGAP

# For PHATE part
from phate import PHATE
import numpy as np
from scipy import sparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize

import graphtools
from sklearn.exceptions import NotFittedError

from sklearn.utils.validation import check_is_fitted

class PHATET(PHATE): 

    def __init__(self, beta = 0.9, **kwargs):
        super(PHATET, self).__init__(**kwargs)

        self.beta = beta

    @property
    def diff_op(self):
        """diff_op :  array-like, shape=[n_samples, n_samples] or [n_landmark, n_landmark]
        The diffusion operator built from the graph
        """
        if self.graph is not None:
            if isinstance(self.graph, graphtools.graphs.LandmarkGraph):
                diff_op = self.graph.landmark_op
            else:
                diff_op = self.graph.diff_op
            if sparse.issparse(diff_op):
                diff_op = diff_op.toarray()

            dim = diff_op.shape[0]

            diff_op_tele = self.beta * diff_op + (1 - self.beta) * 1 / dim * np.ones((dim, dim))


            return diff_op_tele

        else:
            raise NotFittedError(
                "This PHATE instance is not fitted yet. Call "
                "'fit' with appropriate arguments before "
                "using this method."
            )

# Removed MakeRFPHATE redo
# TODO: Redo below class as factory to produce RF-PHATE conditionally for classification/regression
# class RFPHATE(RFProximity, PHATET):

def RFPHATE(prediction_type = None,
            y = None,           
            n_components = 2,
            prox_method = 'rfgap',
            matrix_type = 'sparse',
            n_landmark = 2000,
            t = "auto",
            n_pca = 100,
            mds_solver = "sgd",
            mds_dist = "euclidean",
            mds = "metric",
            n_jobs = 1,
            random_state = None,
            verbose = 0,
            non_zero_diagonal = True,
            beta = 0.9,
            self_similarity = False,
            **kwargs):
    
    # TODO: Not sure how to implement kwargs in this context.
    """An RF-PHATE class which is used to fit a random forest, generate RF-proximities,
       and create RF-PHATE embeddings.

    Parameters
    ----------
    n_components : int
        The number of dimensions for the RF-PHATE embedding

    prox_method : str
        The type of proximity to be constructed.  Options are 'original', 'oob', and
        'rfgap' (default is 'oob')

    matrix_type : str
        Whether the proximity type should be 'sparse' or 'dense' (default is sparse)
    
    n_landmark : int, optional
        number of landmarks to use in fast PHATE (default is 2000)

    t : int, optional
        power to which the diffusion operator is powered.
        This sets the level of diffusion. If 'auto', t is selected
        according to the knee point in the Von Neumann Entropy of
        the diffusion operator (default is 'auto')

    n_pca : int, optional
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time (default is 100)

    mds : string, optional
        choose from ['classic', 'metric', 'nonmetric'].
        Selects which MDS algorithm is used for dimensionality reduction
        (default is 'metric')

    mds_solver : {'sgd', 'smacof'}
        which solver to use for metric MDS. SGD is substantially faster,
        but produces slightly less optimal results (default is 'sgd')

    mds_dist : string, optional
        Distance metric for MDS. Recommended values: 'euclidean' and 'cosine'
        Any metric from `scipy.spatial.distance` can be used. Custom distance
        functions of form `f(x, y) = d` are also accepted (default is 'euclidean')

    n_jobs : integer, optional
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used (default is 1)

    random_state : integer
        random seed state set for RF and MDS


    verbose : int or bool
        If `True` or `> 0`, print status messages (default is 0)

    non_zero_diagonal: bool
        Only used if prox_method == 'rfgap'.  Replaces the zero-diagonal entries
        of the rfgap proximities with ones (default is True)

    self_similarity: bool  
        Only used if prox_method == 'rfgap'. All points are passed down as if OOB. 
        Increases similarity between an observation and itself as well as other
        points of the same class.
    """

    if prediction_type is None and y is None:
        prediction_type = 'classification'
        
    # In rfgap module, rf is defined without arguements
    rfgap = RFGAP(prediction_type = prediction_type, y = y, **kwargs)

    class RFPHATE(rfgap.__class__, PHATET):
    # class RFPHATE(PHATET):
    
        def __init__(
            self,
            n_components = n_components,
            prox_method  = prox_method,
            matrix_type  = matrix_type,
            n_landmark   = n_landmark,
            t            = t,
            n_pca        = n_pca,
            mds_solver   = mds_solver,
            mds_dist     = mds_dist ,
            mds          = mds,
            n_jobs       = n_jobs,
            random_state = random_state,
            verbose      = verbose,
            non_zero_diagonal = non_zero_diagonal,
            beta         = beta,
            self_similarity = self_similarity,
            **kwargs
            ):

            super(RFPHATE, self).__init__(**kwargs)
            
            self.n_components = n_components
            self.t = t
            self.n_landmark = n_landmark
            self.mds = mds
            self.n_pca = n_pca
            self.knn_dist = 'precomputed_affinity'
            self.mds_dist = mds_dist
            self.mds_solver = mds_solver
            self.random_state = random_state
            self.n_jobs = n_jobs

            self.graph = None
            self._diff_potential = None
            self.embedding = None
            self.x = None
            self.optimal_t = None
            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.verbose = verbose
            self.non_zero_diagonal = non_zero_diagonal
            self.beta = beta
            self.self_similarity = self_similarity

        # From https://www.geeksforgeeks.org/class-factories-a-powerful-pattern-in-python/
            for k, v in kwargs.items():
                setattr(self, k, v)
                
                
               
        def _fit(self, x, y, x_test = None):
            
            self.fit(x, y, x_test = x_test)

            if self.prox_method == 'rfgap' and self.self_similarity:
                self.proximity = self.prox_extend(x)
            else:
                self.proximity = self.get_proximities()

                
        # We may not need this afterall
        def _fit_transform_test(self, x, y, x_test = None):
            
            self.fit(x, y, x_test = x_test)
            
            self.test_proximity = self.prox_extend(x_test)
            
            phate_op = PHATET(n_components = self.n_components,
                t = self.t,
                n_landmark = self.n_landmark,
                mds = self.mds,
                n_pca = self.n_pca,
                knn_dist = self.knn_dist,
                mds_dist = self.mds_dist,
                mds_solver = self.mds_solver,
                random_state = self.random_state,
                verbose = self.verbose, 
                beta = self.beta)
            
            self.phate_op = phate_op
            self.embedding_ = phate_op.fit_transform(self.test_proximity)
            
            return self.embedding_
            
            
             
                
                
        def _transform(self, x):
            
            check_is_fitted(self)
            
            phate_op = PHATET(n_components = self.n_components,
                t = self.t,
                n_landmark = self.n_landmark,
                mds = self.mds,
                n_pca = self.n_pca,
                knn_dist = self.knn_dist,
                mds_dist = self.mds_dist,
                mds_solver = self.mds_solver,
                random_state = self.random_state,
                verbose = self.verbose, 
                beta = self.beta)
            
            self.phate_op = phate_op
            self.embedding_ = phate_op.fit_transform(self.proximity)
            
            return self.embedding_
            
            
        

        def _fit_transform(self, x, y, x_test = None):

            """Internal method for fitting and transforming the data
            
            Parameters
            ----------
            x : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Internally, its dtype will be converted to dtype=np.float32.
                If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                The target values (class labels in classification, real numbers in regression).
            """

            n,  _= x.shape
            
            # TODO: not using sample weight
            self.fit(x, y, x_test = x_test)

            if self.prox_method == 'rfgap' and self.self_similarity:
                if x_test is None:
                    proximity = self.prox_extend(x)
                else:
                    proximity = self.prox_extend(np.concatenate([x, x_test]))
            else:
                proximity = self.get_proximities()
                            
            phate_op = PHATET(n_components = self.n_components,
                t = self.t,
                n_landmark = self.n_landmark,
                mds = self.mds,
                n_pca = self.n_pca,
                knn_dist = self.knn_dist,
                mds_dist = self.mds_dist,
                mds_solver = self.mds_solver,
                random_state = self.random_state,
                verbose = self.verbose, 
                beta = self.beta)
            
            self.phate_op = phate_op

            self.embedding_ = phate_op.fit_transform(proximity)
            self.proximity = proximity # Wasn't previously saved, added 10.31.2023

        def fit_transform(self, x, y, x_test = None):

            """Applies _fit_tranform to the data, x, y, and returns the RF-PHATE embedding

            x : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Internally, its dtype will be converted to dtype=np.float32.
                If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                The target values (class labels in classification, real numbers in regression).


            Returns
            -------
            array-like (n_features, n_components)
                A lower-dimensional representation of the data following the RF-PHATE algorithm
            """
            self._fit_transform(x, y, x_test)
            return self.embedding_
        
        def extend_to_data(self, data):
            """Build transition matrix from new data to the training graph (Full or Landmark)

            Creates a transition matrix such that `Y` can be approximated by
            a linear combination of landmarks. Any
            transformation of the landmarks can be trivially applied to `Y` by
            performing

            `transform_Y = transitions.dot(transform)`

            Parameters
            ----------

            Y: array-like, [n_samples_y, n_features]
                new data for which an affinity matrix is calculated
                to the existing data. `n_features` must match
                either the ambient or PCA dimensions

            Returns
            -------

            transitions : array-like, [n_samples_y, self.data.shape[0]]
                Transition matrix from `Y` to `self.data`
            """
            kernel = self.prox_extend(data)
            if isinstance(self.phate_op.graph, graphtools.graphs.LandmarkGraph):
                pnm = sparse.hstack(
                    [
                        sparse.csr_matrix(kernel[:, self.phate_op.graph.clusters == i].sum(axis=1))
                        for i in np.unique(self.phate_op.graph.clusters)
                    ]
                )
                pnm = normalize(pnm, norm="l1", axis=1)
            else:
                pnm = normalize(kernel, norm="l1", axis=1)
            return pnm
    

        def transform(self, data, **kwargs):
            """Basic extension for new points in the embedding space"""
            check_is_fitted(self)
            pnm = self.extend_to_data(data)
            return self.phate_op.graph.interpolate(self.phate_op.embedding, pnm)
    

    return RFPHATE(    
                n_components = n_components,
                prox_method = prox_method,
                matrix_type = matrix_type,
                n_landmark = n_landmark,
                t = t,
                n_pca = n_pca,
                mds_solver = mds_solver,
                mds_dist = mds_dist,
                mds = mds,
                n_jobs = n_jobs,
                random_state = random_state,
                verbose = verbose,
                non_zero_diagonal = non_zero_diagonal,
                beta = beta,
                self_similarity = self_similarity,
                **kwargs)

    # return RFPHATE