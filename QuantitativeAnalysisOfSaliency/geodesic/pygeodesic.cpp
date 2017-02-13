#include <Python.h>
#include "geodesic_algorithm_exact.h"

// define PyInt_* macros for Python 3.x
#ifndef PyInt_Check
#define PyInt_Check             PyLong_Check
#define PyInt_FromLong          PyLong_FromLong
#define PyInt_AsLong            PyLong_AsLong
#define PyInt_Type              PyLong_Type
#endif

static PyObject *compute_geodesic(PyObject *self, PyObject *args, PyObject *keywds)
{
  static char *kwlist[] = {"points", "faces", "sources", "limit",
                           NULL};

  PyObject *py_points, *py_faces, *py_sources, *py_limit;
  std::vector<double> points; 
  std::vector<unsigned> faces, sources;
  float limit;
  std::vector<double> distances;
  PyObject *item;
  unsigned i;

  if(!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO", kwlist,
                                  &py_points, &py_faces, &py_sources, &py_limit))
     return NULL;

  if(!PySequence_Check(py_points)) {
    PyErr_SetString(PyExc_TypeError, "points must be a sequence");
    return NULL;
  }

  if(!PySequence_Check(py_faces)) {
    PyErr_SetString(PyExc_TypeError, "faces must be a sequence");
    return NULL;
  }

  if(!PySequence_Check(py_sources)) {
    PyErr_SetString(PyExc_TypeError, "sources must be a sequence");
    return NULL;
  }

  if(!PyFloat_Check(py_limit)) {
    PyErr_SetString(PyExc_TypeError, "limit must be a float");
    return NULL;
  } else {
    limit = PyFloat_AsDouble(py_limit);
  }

  unsigned points_size = PySequence_Size(py_points);
  unsigned faces_size = PySequence_Size(py_faces);
  unsigned sources_size = PySequence_Size(py_sources);
  points.reserve(points_size);
  faces.reserve(faces_size);
  sources.reserve(sources_size);

  for(i = 0; i < points_size; i ++) {
    item = PySequence_GetItem(py_points, i);
    if(!PyFloat_Check(item)) {
      Py_DECREF(item);
      PyErr_SetString(PyExc_TypeError, "points should be a sequence of floats");
      return NULL;
    }
    points.push_back(PyFloat_AsDouble(item));
    Py_DECREF(item);
  }

  for(i = 0; i < faces_size; i ++) {
    item = PySequence_GetItem(py_faces, i);
    if(!PyInt_Check(item)) {
      Py_DECREF(item);
      PyErr_SetString(PyExc_TypeError, "faces should be a sequence of integers");
      return NULL;
    }
    faces.push_back(PyFloat_AsDouble(item));
    Py_DECREF(item);
  }

  for(i = 0; i < sources_size; i ++) {
    item = PySequence_GetItem(py_sources, i);
    if(!PyInt_Check(item)) {
      Py_DECREF(item);
      PyErr_SetString(PyExc_TypeError, "faces should be a sequence of integers");
      return NULL;
    }
    sources.push_back(PyFloat_AsDouble(item));
    Py_DECREF(item);
  }

  geodesic::Mesh mesh;
  mesh.initialize_mesh_data(points, faces);
  geodesic::GeodesicAlgorithmExact algorithm(&mesh);

  distances.reserve(sources_size*mesh.vertices().size());
  for(unsigned j=0; j<sources_size; ++j) {
    unsigned source_vertex_index = sources[j];
    // std::cout << "source: " << source_vertex_index << "\n";
    geodesic::SurfacePoint source(&mesh.vertices()[source_vertex_index]); 
    std::vector<geodesic::SurfacePoint> all_sources(1,source);
    algorithm.propagate(all_sources, limit);

    for(i=0; i<mesh.vertices().size(); ++i)
    {
      geodesic::SurfacePoint p(&mesh.vertices()[i]);    
      double distance;
      unsigned best_source = algorithm.best_source(p,distance);   //for a given surface point, find closets source and distance to this source
      distances.push_back(distance);
    }
    // std::cout << "source: " << source_vertex_index << ": " << distances.size() << "\n";
  }

  // std::cout << "source: " << distances.size() << "\n";
  PyObject *py_distances = PyList_New(distances.size());
  for (i = 0; i < distances.size(); ++i) {
    PyList_SetItem(py_distances, i, Py_BuildValue("d", distances[i]));
  }
  // std::cout << "source: " << PyList_Size(py_distances) << "\n";

  return py_distances;
}

static PyMethodDef functions[] = {
    {"geodesic", (PyCFunction)compute_geodesic, METH_VARARGS | METH_KEYWORDS,
     "Compute Geodesic Distances."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
//-----------------------------------------------------------------------------
//   Declaration of module definition for Python 3.x.
//-----------------------------------------------------------------------------
static struct PyModuleDef g_ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "geodesic",
    NULL,
    -1,
    functions,                             // methods
    NULL,                                  // m_reload
    NULL,                                  // traverse
    NULL,                                  // clear
    NULL                                   // free
};
#endif



#if PY_MAJOR_VERSION >= 3
PyObject * PyInit_geodesic(void)
{
  return PyModule_Create(&g_ModuleDef);
#else
PyMODINIT_FUNC initgeodesic(void)
{
    Py_InitModule(
        "geodesic", functions
        );
#endif
}