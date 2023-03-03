import pandas as pd
import numpy as np

class FlatModel:
    def __init__(self, data):
        self.flat = self._to_series(data)

    @staticmethod
    def _to_series(data):
        """Converts dict into pd.Series and changes type of each value.

        Keyword arguments:
        data -- dict object of form values input.
        
        Returns:
        series -- pd.Series object.
        """
        series = pd.Series(data)
        types = [int, float, float, float, 
                 int, int, str, int,
                 str, str, str, str, str]
        columns = series.index.to_list()

        for i, column in enumerate(columns):
            series[column] = types[i](series[column])

        return series


    def validate_data(self):
        """Check if 'flat' parameter has correct values.

        Returns:
        error -- None if values are correct, str if not.
        """
        error = None
        if self.flat.living_area + self.flat.kitchen_area > self.flat.all_area:
            error = 'Данные площади введены неверно, попробуйте ещё раз'
        elif self.flat.floor > self.flat.floors_count:
            error = 'Этаж введён неверно, попробуйте ещё раз'

        return error


    @staticmethod
    def _count_house_age(data):
        '''Transform "fondation_year" into "house_age" column.

        Keyword arguments:
        data -- pd.Series object with "fondation_year".

        Returns:
        data -- pd.Series object with "house_age" column.
        '''
        data['house_age'] = 2023 - data['fondation_year']
        data = data.drop(['fondation_year'])

        return data


    @staticmethod
    def _set_balcony(data, result):
        '''Encode column into several new feature-columns.

        Keyword arguments:
        data -- pd.Series object with needed values.
        result -- pd.Series object with pre-set columns.

        Returns:
        result -- pd.Series object with pre-set columns filled with values.
        '''
        match data['balcony_loggia']:
            case '1 балкон':
                result['Балкон'] = 1
            case '2 балкона':
                result['Балкон'] = 2
            case '1 лоджия':
                result['Лоджия'] = 1
            case '2 лоджии':
                result['Лоджия'] = 2
            case '1 балкон, 1 лоджия':
                result[['Балкон', 'Лоджия']] = 1

        return result


    @staticmethod
    def _set_window_view(data, result):
        '''Encode column into several new feature-columns.

        Keyword arguments:
        data -- pd.Series object with needed values.
        result -- pd.Series object with pre-set columns.

        Returns:
        result -- pd.Series object with pre-set columns filled with values.
        '''
        match data['window_view']:
            case 'Во двор':
                result['Во двор'] = 1
            case 'На улицу':
                result['На улицу'] = 1
            case 'На улицу и двор':
                result[['Во двор', 'На улицу']] = 1

        return result


    def transform(self):
        '''Transform input pd.Series object into 
        encoded and log-scaled numpy.array.

        Returns:
        result -- list-like numpy.array object.
        '''
        self._count_house_age(self.flat)
        num_feats = [
            'room_count', 'all_area', 
            'living_area', 'kitchen_area', 'floor', 
            'floors_count', 'house_age'
        ]
        scale_feats = [
            'room_count', 'all_area', 
            'living_area', 'kitchen_area', 'house_age'
            ]
        feats = [
            'room_count', 'all_area', 'house_age', 'На улицу',
            'Во двор', 'Лоджия', 'Балкон', 'Совмещенный', 'Раздельный',
            'Дизайнерский', 'Евроремонт', 'Без ремонта', 
            'Косметический', 'Новостройка', 'Вторичка', 
            'Перекресток', 'ИП Бобынцева Л.Н.', 
            'Перспектива 24 Курск', 'КУРСКАЯ НЕДВИЖИМОСТЬ', 
            'ProДом46', 'Мой город', 'НДС недвижимость',
            'Эпсилон-недвижимость', 'Собственник', 'Риелтор',
            'floors_count', 'floor',  'kitchen_area', 
            'living_area', 'Другое агентство'
        ]
        result = pd.Series(data=0, index=feats)
        result[num_feats] = self.flat[num_feats].values
        result = self._set_balcony(self.flat, result)
        result = self._set_window_view(self.flat, result)
        result[self.flat['renovation_type']] = 1
        result[self.flat['contact_type']] = 1
        result[self.flat['housing_type']] = 1
        # Log(x + 1) scale.
        result[scale_feats] = np.log1p(pd.to_numeric(result[scale_feats]))
        result = result.values.reshape(1, -1)

        return result