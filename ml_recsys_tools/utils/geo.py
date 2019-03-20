import pprint
import io

import numpy as np
import os
import gmaps
import ipywidgets.embed
import colorsys


class ItemsGeoMap:

    def __init__(self, lat_col='lat', long_col='long'):
        self.lat_col = lat_col
        self.long_col = long_col
        self.fig = None
        gmaps.configure(api_key=os.environ['GOOGLE_MAPS_API_KEY'])

    @property
    def loc_cols(self):
        return [self.lat_col, self.long_col]

    def reset_map(self):
        self.fig = None

    def _check_get_view_fig(self):
        if self.fig is None:
            self.fig = gmaps.figure()

    def add_heatmap(self,
                    df_items,
                    color=(0, 250, 50),
                    opacity=0.6,
                    sensitivity=5,
                    spread=30, ):

        self._check_get_view_fig()

        self.fig.add_layer(
            gmaps.heatmap_layer(
                df_items[self.loc_cols].values,
                opacity=opacity,
                max_intensity=sensitivity,
                point_radius=spread,
                dissipating=True,
                gradient=[list(color) + [0],
                          list(color) + [1]]))
        return self

    def add_markers(self,
                    df_items,
                    max_markers=1000,
                    color='red',
                    size=2,
                    opacity=1.0,
                    fill=True
                    ):

        self._check_get_view_fig()

        marker_locs, marker_info = self._markers_with_info(df_items, max_markers=max_markers)

        self.fig.add_layer(gmaps.symbol_layer(
            marker_locs[self.loc_cols].values,
            fill_color=color,
            stroke_color=color,
            fill_opacity=opacity if fill else 0,
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

    def draw_items(self, df_items, **kwargs):
        self.add_heatmap(df_items, **kwargs)
        self.add_markers(df_items, **kwargs)
        return self

    def write_html(self, *, path=None, map_height=800, **kwargs):
        """
        writes map to html string or file.
        :param path: path to file (optional). if None the methods returns the html string.
        :param map_height: height of map element
        :return: if path is None return the html string, if path is not None returns the path (file mode).
        """
        for w in self.fig.widgets.values():
            if isinstance(w, ipywidgets.Layout) and str(w.height).endswith('px'):
                w.height = f'{map_height}px'

        if path:
            ipywidgets.embed.embed_minimal_html(fp=path, views=[self.fig], **kwargs)
            return path
        else:
            with io.StringIO() as fp:
                ipywidgets.embed.embed_minimal_html(fp=fp, views=[self.fig], **kwargs)
                fp.flush()
                return fp.getvalue()

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


class PropertyGeoMap(ItemsGeoMap):

    _score_key = '_score'

    def __init__(self, link_base_url='www.domain.com.au', **kwargs):
        super().__init__(**kwargs)
        self.site_url = link_base_url

    def _markers_with_info(self, df_items, max_markers):
        marker_locs = df_items.iloc[:max_markers]
        marker_info = []
        for _, item_data in marker_locs.iterrows():
            item = item_data.to_dict()
            url = f"https://{self.site_url}/{item.get('property_id')}"
            marker_info.append(
                f"""     
                <dl><a style="font-size: 16px" href='{url}' target='_blank'>{url}</a><dt> 
                score: {item.get(self._score_key, np.nan) :.2f} | {item.get('price')} $ 
                {item.get('property_type')} ({item.get('buy_or_rent')}) | {item.get('bedrooms')} B ' \
                '| {item.get('bathrooms')} T | {item.get('carspaces')} P <br />  ' \
                '{item.get('land_area')} Sqm | in {item.get('suburb')} | with {item.get('features_list')}'</dt></dl>
                """)
        return marker_locs, marker_info


