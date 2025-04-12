=====================
Browser library 1.5.0
=====================


.. default-role:: code


Browser_ is a web testing library for `Robot Framework`_ that utilizes
the Playwright_ tool internally. Browser library 1.5.0 is a new release with
**UPDATE** enhancements and bug fixes. **ADD more intro stuff...**
**REMOVE this section with final releases or otherwise if release notes contain
all issues.**
All issues targeted for Browser library v1.5.0 can be found
from the `issue tracker`_.
**REMOVE ``--pre`` from the next command with final releases.**
If you have pip_ installed, just run
::
   pip install --pre --upgrade robotframework-browser
   rfbrowser init
to install the latest available release or use
::
   pip install robotframework-browser==1.5.0
   rfbrowser init
to install exactly this version. Alternatively you can download the source
distribution from PyPI_ and install it manually.
Browser library 1.5.0 was released on Tuesday October 13, 2020. Browser supports
Python **ADD VERSIONS**, Playwright **ADD VERSIONS** and
Robot Framework **ADD VERSIONS**.
.. _Robot Framework: http://robotframework.org
.. _Browser: https://github.com/MarketSquare/robotframework-browser
.. _Selenium: https://github.com/microsoft/playwright
.. _pip: http://pip-installer.org
.. _PyPI: https://pypi.python.org/pypi/robotframework-browser
.. _issue tracker: https://github.com/MarketSquare/robotframework-browser/milestones%3Av1.5.0


.. contents::
   :depth: 2
   :local:

Most important enhancements
===========================

**EXPLAIN** or remove these.

- Switch Page  NEW  = to page that has been created after the current page (`#413`_)
- Wait for request and response matchers (regexp at least) do not seem to work (`#230`_)
- Update geolocation keyword (`#226`_)

Full list of fixes and enhancements
===================================

.. list-table::
    :header-rows: 1

    * - ID
      - Type
      - Priority
      - Summary
    * - `#413`_
      - bug
      - critical
      - Switch Page  NEW  = to page that has been created after the current page
    * - `#230`_
      - bug
      - high
      - Wait for request and response matchers (regexp at least) do not seem to work
    * - `#226`_
      - enhancement
      - high
      - Update geolocation keyword
    * - `#403`_
      - bug
      - medium
      - Improve exception raised by library when trying to click element which can not be clicked.

Altogether 4 issues. View on the `issue tracker <https://github.com/MarketSquare/robotframework-browser/issues?q=milestone%3Av1.5.0>`__.

.. _#413: https://github.com/MarketSquare/robotframework-browser/issues/413
.. _#230: https://github.com/MarketSquare/robotframework-browser/issues/230
.. _#226: https://github.com/MarketSquare/robotframework-browser/issues/226
.. _#403: https://github.com/MarketSquare/robotframework-browser/issues/403
