.. RSS documentation master file, created by
   sphinx-quickstart on Sat Sep  3 14:06:50 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NIST RSoXS Simulation Suite (NRSS)
======================================

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started/idx_getting_started
   user_guide/idx_user_guide
   api/idx_api
   development/idx_development

.. |LogoLarge| image:: /_static/Logo_NRSS.svg
   :class: dark-light
   :alt: NRSS Logo
   :width: 300

.. grid:: 2

   .. grid-item::
      :columns: 3

      |LogoLarge|

   .. grid-item::
      :columns: 9
      :child-align: center

      The NIST RSoXS Simulation Suite (NRSS) is a collection of Python and C++/CUDA tools for 
      creating, simulating, and analyzing Resonant Soft X-ray Scattering (RSoXS). It leverages 
      the `CyRSoXS <https://github.com/usnistgov/cyrsoxs>`_ C++/CUDA library for GPU-accelerated 
      simulations, and the `PyHyperScattering <https://github.com/usnistgov/PyHyperScattering>`_ 
      library for reducing and analyzing the simulated scattering patterns. For more information 
      on what RSoXS is and how you can possibly apply it in your own research, check out the `NIST 
      RSoXS project page <https://www.nist.gov/programs-projects/resonant-soft-x-ray-scattering-rsoxs>`_

Documentation
==================

.. list-table:: 
   :widths: 30 30
   :header-rows: 0

   * - :ref:`Getting Started <Getting_Started>` Package Installation and a Start-to-Finish Tutorial. Beginners should start here. 
     - :ref:`User Guide <User_Guide>` How-To Guides (recipes) for specific data reduction, analysis, and visualization tasks.
   * - :ref:`Software Reference <Reference_API>` A summary of the CyRSoXS Simulation Model and the NRSS API Documentation. 
     - :ref:`Development <Development>` Information and resources regarding the scope and development philosophy of this project, along with information on contributing and licensing.

.. warning ::
   NIST Disclaimer: Any identification of commercial or open-source software in this document is done so purely in 
   order to specify the methodology adequately. Such identification is not intended to imply 
   recommendation or endorsement by the National Institute of Standards and Technology, nor is it 
   intended to imply that the softwares identified are necessarily the best available for the purpose.


Sitemap
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



