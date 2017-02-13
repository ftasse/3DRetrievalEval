from distutils.core import setup, Extension

setup(
    name="geodesic",
    ext_modules = [Extension("geodesic", ["pygeodesic.cpp"], extra_compile_args = ['-O3'])],
    author = 'Flora Tasse',
    author_email = 'fp289@cl.cam.ac.uk',
    description = ("Python wrapper around Microsoft Geodesic Distance implementation (http://research.microsoft.com/en-us/um/people/hoppe/proj/geodesics/)"),
    license = 'GPL v3',
    keywords = 'geodesic distance',
    
    )