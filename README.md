# module-code-examples
Example codes for each module.

Main branch contains the latest state of the course.
For each week, we have a separate branch that reflects the course state of the week.

This is the branch for week 1: 14-20 October.
- The lecture is from 16 October 16:00-18:00 CET.


Example of uploading package to the volume:

'''
databricks auth login --host HOST
uv build
databricks fs cp dist/wheel dbfs:/Volumes/my_catalog/my_schema/my_volume/
'''

