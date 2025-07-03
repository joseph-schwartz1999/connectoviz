API Reference
=============

This section details the public API of ConnectoViz, documenting all discovered modules and their members.

..
   The `.. automodule:: connectoviz` directive below will document anything directly
   defined in your connectoviz/__init__.py or explicitly imported into its __all__.
   It does NOT automatically recurse into subpackages. To document subpackages,
   you need separate automodule directives for them.

.. automodule:: connectoviz
   :members:        # Documents anything directly in connectoviz/__init__.py
   :undoc-members:  # Include if you want undocumented members listed
   :show-inheritance:

..
   Now, explicitly document the submodules and modules that contain your code.
   Each of these will create a heading and list all members within them.

connectoviz.plotting Module
---------------------------

.. automodule:: connectoviz.plotting.circular_plots
   :members:        # Documents CircularPlotBuilder, plot_circular_connectome, etc.
   :undoc-members:
   :show-inheritance:

connectoviz.core Module
-----------------------

.. automodule:: connectoviz.core.connectome
   :members:        # Documents anything defined in connectome.py
   :undoc-members:
   :show-inheritance:

..
   You can add more automodule directives for other specific modules/subpackages
   you create in the future, e.g., connectoviz.data.your_data_module

API Summaries
-------------

Here is a summary of key components within the ConnectoViz API:

.. autosummary::
   :toctree: _autosummary
   :nosignatures: # Optional: do not show the signature in the summary table itself

   # You can list modules here to create summary links to their main page:
   connectoviz.plotting.circular_plots
   connectoviz.core.connectome

   # Or, list specific key classes/functions you want direct summary links for:
   connectoviz.visualization.circular_plot_builder.CircularPlotBuilder
   # connectoviz.plotting.circular_plots.plot_circular_connectome # If it's a function directly in circular_plots.py
   # connectoviz.core.connectome.YourMainConnectomeClass