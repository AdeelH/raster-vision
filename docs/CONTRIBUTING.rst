Contributing
============

We are happy to take contributions! It is best to get in touch with the maintainers about larger features or design changes *before* starting the work, as it will make the process of accepting changes smoother.

Contributor License Agreement (CLA)
-----------------------------------

Everyone who contributes code to Raster Vision will be asked to sign a CLA, which is based off of the Apache CLA.

1. Download a copy of **one of** the following:

   - Individual Contributor: `Raster Vision Individual Contributor License Agreement <_static/cla/2024_10_10-Raster-Vision-Open-Source-Contributor-Agreement-Individual.pdf>`_ 
   - Corporate Contributor: `Raster Vision Corporate Contributor License Agreement <_static/cla/2024_10_10-Raster-Vision-Open-Source-Contributor-Agreement-Corporate.pdf>`_

2. Sign the CLA -- either physically on a printout or digitally using appropriate PDF software.

3. Send the signed CLAs to Element 84 via **one of** the following methods:

   - Emailing the document to contracts@element84.com
   - Mailing a hardcopy to: ``Element 84, 210 N. Lee Street Suite 203 Alexandria, VA 22314, USA``.

Code style
----------

Raster Vision uses `ruff <https://docs.astral.sh/ruff/>`__ for formatting and linting, and `pyright <https://microsoft.github.io/pyright/>`__ for type checking. Their configuration is in the root ``pyproject.toml``. Both are installed by ``scripts/setup_dev_env``.

- Format the code and sort imports with ``scripts/format_code`` (``ruff check --select I --fix`` and ``ruff format``). Files whose import order matters are marked with ``# isort: skip_file`` (or ``# isort: off``/``# isort: on`` around the relevant imports).
- Check the code with ``scripts/style_tests``, which runs ``ruff check``, ``ruff format --check`` and ``pyright``. CI runs the same checks.

Tests
-----

Run the unit tests with ``scripts/unit_tests`` (or ``python -m unittest discover -t . tests``; ``pytest tests`` also works) and the integration tests with ``scripts/integration_tests``. Neither requires the Docker image.


.. toctree::
    :maxdepth: 2
    :hidden:

    self
    release
    Code of Conduct <CODE_OF_CONDUCT>
