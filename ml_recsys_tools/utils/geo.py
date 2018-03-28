import pprint
import numpy as np
import os
import gmaps
import ipywidgets.embed
import colorsys


class ItemsGeoMapper:

    def __init__(self, df_items=None, lat_col='lat', long_col='long'):
        self.df_items = df_items
        self.lat_col = lat_col
        self.long_col = long_col
        self.fig = None
        gmaps.configure(api_key=os.environ['GOOGLE_MAPS_API_KEY'])

    @property
    def loc_cols(self):
        return [self.lat_col, self.long_col]

    def _ceter_location(self):
        return self.df_items[self.loc_cols].apply(np.mean).tolist()

    def _zoom_heuristic(self):

        # https://stackoverflow.com/questions/6048975/google-maps-v3-how-to-calculate-the-zoom-level-for-a-given-bounds
        def gm_heuristic(min_deg, max_deg):
            if max_deg - min_deg > 0:
                return int(np.log(1000 * 360 / (max_deg - min_deg) / 256) / np.log(2))
            else:
                return 14

        ranges = self.df_items[self.loc_cols].apply([np.min, np.max]).as_matrix()

        zoom_level_lat = gm_heuristic(ranges[0][0], ranges[1][0])

        zoom_level_long = gm_heuristic(ranges[0][1], ranges[1][1])

        return min(zoom_level_lat, zoom_level_long)

    def _check_get_view_fig(self):
        if self.fig is None:
            self.fig = gmaps.figure()
            self.fig.widgets.clear()  # clear any history
            self.fig = gmaps.figure(center=self._ceter_location(), zoom_level=self._zoom_heuristic())

    def add_heatmap(self,
                    df_items=None,
                    color=(0, 250, 50),
                    opacity=0.6,
                    sensitivity=5,
                    spread=30, ):

        self._check_get_view_fig()

        if df_items is None:
            df_items = self.df_items

        self.fig.add_layer(
            gmaps.heatmap_layer(
                df_items[self.loc_cols].as_matrix(),
                opacity=opacity,
                max_intensity=sensitivity,
                point_radius=spread,
                dissipating=True,
                gradient=[list(color) + [0],
                          list(color) + [1]]))
        return self

    def add_markers(self,
                    df_items=None,
                    max_markers=1000,
                    color='red',
                    size=2,
                    opacity=1.0,
                    ):

        self._check_get_view_fig()

        if df_items is None:
            df_items = self.df_items

        marker_locs, marker_info = self._markers_with_info(df_items, max_markers=max_markers)

        self.fig.add_layer(gmaps.symbol_layer(
            marker_locs[self.loc_cols].as_matrix(),
            fill_color=color,
            stroke_color=color,
            fill_opacity=opacity,
            stroke_opacity=opacity,
            scale=size,
            info_box_content=marker_info))
        return self

    @staticmethod
    def _markers_with_info(df_items, max_markers):
        marker_locs = df_items.iloc[:max_markers]
        info_box_template = \
            """
            <dl>            
            <dt>{description}</dt>
            </dl>
            """
        marker_info = [
            info_box_template.format(
                description=pprint.pformat(item_data.to_dict()))
            for _, item_data in marker_locs.iterrows()]

        return marker_locs, marker_info

    def draw_listings(self, df_items=None, **kwargs):
        self.add_heatmap(df_items, **kwargs)
        self.add_markers(df_items, **kwargs)
        return self

    def write_html(self, path):
        for w in self.fig.widgets.values():
            if isinstance(w, ipywidgets.Layout) and w.height == '400px':
                w.height = '700px'

        ipywidgets.embed.embed_minimal_html(path, views=[self.fig])
        return self

    @staticmethod
    def random_color():
        return tuple(map(int, np.random.randint(0, 255, 3)))  # because of bug in gmaps type checking

    @staticmethod
    def get_n_spaced_colors(n):
        # max_value = 16581375  # 255**3
        # interval = int(max_value / n)
        # colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
        # return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

        HSV_tuples = [(x * 1.0 / n, 1.0, 0.8) for x in range(n)]
        RGB_tuples = list(map(lambda x:
                              tuple(list(map(lambda f: int(f * 255),
                                             colorsys.hsv_to_rgb(*x)))),
                              HSV_tuples))
        return RGB_tuples
