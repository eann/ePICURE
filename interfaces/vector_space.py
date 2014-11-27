import numpy as np

class VectorSpace:
    """An abstract python interface used to describe *one dimensional
    functions* on `[a,b]`, as *coefficients* times *basis functions*, with
    access to single basis functions, their derivatives, their support and
    the  splitting of `[a,b]` into sub-intervals where *each basis* is
    assumed to be smooth.

    The base class constructs the constant vector space.
    """
    def __init__(self, a=0.0, b=1.0):
        """ Pure interface class. It generates the constant on [0,1]. """
        self.n_dofs = 1
        self.n_dofs_per_end_point = 0
        self.n_cells = 1
        self.cells = np.array([a, b])

    def check_index(self, i):
        assert i< self.n_dofs, \
            'Trying to access base %, but we only have %'.format(i, self.n_dofs)

    def basis(self, i):
        self.check_index(i)
        return lambda x: 1.0

    def basis_der(self, i, d):
        self.check_index(i)
        return lambda x: 0.0

    def basis_span(self, i):
        """ VectorSpace.basis_span(i): a tuple indicating the start and end indices into the cells object where the i-th basis function is different from zero"""
        self.check_index(i)
        return (0, 1)

    def cell_span(self, i):
        """ An array of indices containing the basis functions which are non zero on the i-th cell """
        self.check_index(i)
        return [0]

    def eval(self, c, x):
		"""eval(c,x)
			Evaluates vector c (given as a list of basis coefficients) at point x."""
        assert len(c) == self.n_dofs, \
            'Incompatible vector. It should have length %. It has lenght %'.format(self.n_dofs, len(c))
        # Find the cell where x lies:
        y = 0*x
        # TBD: make this function aware of locality
        for i in xrange(self.n_dofs):
            y += self.basis(i)(x)*c[i]
    
    def element(self, c):
        """  VectorSpace.element(c): a callable function, representing sum(c[i] * basis[i]), which exploits the locality of the basis functions """
        assert len(c) == self.n_dofs, \
            'Incompatible vector. It should have length %. It has lenght %'.format(self.n_dofs, len(c))
        return lambda x: self.eval(c, x)
