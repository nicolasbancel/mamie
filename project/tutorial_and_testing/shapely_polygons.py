# Taken from https://stackoverflow.com/questions/39338550/cut-a-polygon-with-two-lines-in-shapely


from shapely.ops import linemerge, unary_union, polygonize
from shapely.geometry import LineString, Polygon

# Define the Polygon and the cutting line
line = LineString([(-5, -5), (5, 5)])
polygon = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])


def cut_polygon_by_line(polygon, line):
    merged = linemerge([polygon.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)


def plot(shapely_objects, figure_path="fig.png"):
    from matplotlib import pyplot as plt
    import geopandas as gpd

    boundary = gpd.GeoSeries(shapely_objects)
    boundary.plot(color=["red", "green", "blue", "yellow", "yellow"])
    plt.savefig(figure_path)


result = cut_polygon_by_line(polygon, line)
print(result)
plot(result)
print(result[0].intersection(result[1]))
