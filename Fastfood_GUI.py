import tkinter as tk
from fastfood_model import *

class Fastfood_GUI:
    def __init__(self):

        # Create the main window.
        self.main_window = tk.Tk()
        self.main_window.title("Restaurant Predictor")

        # Create two frames to group widgets.
        self.one_frame = tk.Frame()
        self.three_frame = tk.Frame()
        self.four_frame = tk.Frame()
        self.five_frame = tk.Frame()
        self.six_frame = tk.Frame()
        self.seven_frame = tk.Frame()
        self.eight_frame = tk.Frame()
        self.nine_frame = tk.Frame()
        self.ten_frame = tk.Frame()
        self.eleven_frame = tk.Frame()
        self.twelve_frame = tk.Frame()
        self.thirteen_frame = tk.Frame()
        self.fourteen_frame = tk.Frame()
        self.fifteen_frame = tk.Frame()
        self.sixteen_frame = tk.Frame()
        self.seventeen_frame = tk.Frame()
        self.eighteen_frame = tk.Frame()
        self.nineteeen_frame = tk.Frame()

        self.title_label = tk.Label(self.one_frame, text='RESTAURANT PREDICTOR',fg="Blue", font=("Helvetica", 18))
        self.title_label.pack()

        self.item_label = tk.Label(self.three_frame, text='item:')
        self.item_entry = tk.Entry(self.three_frame, bg="white", fg="black", width=10)
        self.item_label.pack(side='left')
        self.item_entry.pack(side='left')

        self.calories_label = tk.Label(self.four_frame, text='calories:')
        self.calories_entry = tk.Entry(self.four_frame, bg="white", fg="black", width=10)
        self.calories_label.pack(side='left')
        self.calories_entry.pack(side='left')

        self.cal_fat_label = tk.Label(self.five_frame, text='calories from fat:')
        self.cal_fat_entry = tk.Entry(self.five_frame, bg="white", fg="black", width=10)
        self.cal_fat_label.pack(side='left')
        self.cal_fat_entry.pack(side='left')

        self.total_fat_label = tk.Label(self.six_frame, text='total fat:')
        self.total_fat_entry = tk.Entry(self.six_frame, bg="white", fg="black", width=10)
        self.total_fat_label.pack(side='left')
        self.total_fat_entry.pack(side='left')

        self.sat_fat_label = tk.Label(self.seven_frame, text='Saturated fat:')
        self.sat_fat_entry = tk.Entry(self.seven_frame, bg="white", fg="black", width=10)
        self.sat_fat_label.pack(side='left')
        self.sat_fat_entry.pack(side='left')

        self.trans_fat_label = tk.Label(self.eight_frame, text='Trans fat:')
        self.trans_fat_entry = tk.Entry(self.eight_frame, bg="white", fg="black", width=10)
        self.trans_fat_label.pack(side='left')
        self.trans_fat_entry.pack(side='left')

        self.cholesterol_label = tk.Label(self.nine_frame, text='cholesterol:')
        self.cholesterol_entry = tk.Entry(self.nine_frame, bg="white", fg="black", width=10)
        self.cholesterol_label.pack(side='left')
        self.cholesterol_entry.pack(side='left')

        self.sodium_label = tk.Label(self.ten_frame, text='sodium:')
        self.sodium_entry = tk.Entry(self.ten_frame, bg="white", fg="black", width=10)
        self.sodium_label.pack(side='left')
        self.sodium_entry.pack(side='left')

        self.total_carb_label = tk.Label(self.eleven_frame, text='total carb:')
        self.total_carb_entry = tk.Entry(self.eleven_frame, bg="white", fg="black", width=10)
        self.total_carb_label.pack(side='left')
        self.total_carb_entry.pack(side='left')

        self.fiber_label = tk.Label(self.twelve_frame, text='fiber:')
        self.fiber_entry = tk.Entry(self.twelve_frame, bg="white", fg="black", width=10)
        self.fiber_label.pack(side='left')
        self.fiber_entry.pack(side='left')

        self.sugar_label = tk.Label(self.thirteen_frame, text='sugar:')
        self.sugar_entry = tk.Entry(self.thirteen_frame, bg="white", fg="black", width=10)
        self.sugar_label.pack(side='left')
        self.sugar_entry.pack(side='left')

        self.protein_label = tk.Label(self.fourteen_frame, text='protein:')
        self.protein_entry = tk.Entry(self.fourteen_frame, bg="white", fg="black", width=10)
        self.protein_label.pack(side='left')
        self.protein_entry.pack(side='left')

        self.vit_a_label = tk.Label(self.fifteen_frame, text='Vitamin A:')
        self.vit_a_entry = tk.Entry(self.fifteen_frame, bg="white", fg="black", width=10)
        self.vit_a_label.pack(side='left')
        self.vit_a_entry.pack(side='left')

        self.vit_c_label = tk.Label(self.sixteen_frame, text='Vitamin C:')
        self.vit_c_entry = tk.Entry(self.sixteen_frame, bg="white", fg="black", width=10)
        self.vit_c_label.pack(side='left')
        self.vit_c_entry.pack(side='left')

        self.calcium_label = tk.Label(self.seventeen_frame, text='Calcium:')
        self.calcium_entry = tk.Entry(self.seventeen_frame, bg="white", fg="black", width=10)
        self.calcium_label.pack(side='left')
        self.calcium_entry.pack(side='left')

        self.salad_label = tk.Label(self.eighteen_frame, text='salad:')
        self.salad_entry = tk.Entry(self.eighteen_frame, bg="white", fg="black", width=10)
        self.salad_label.pack(side='left')
        self.salad_entry.pack(side='left')

        self.salad_label = tk.Label(self.eighteen_frame, text='salad:')
        self.salad_entry = tk.Entry(self.eighteen_frame, bg="white", fg="black", width=10)
        self.salad_label.pack(side='left')
        self.salad_entry.pack(side='left')

        self.hd_predict_ta = tk.Text(self.nineteeen_frame,height = 10, width = 25,bg= 'light blue')

        self.btn_predict = tk.Button(self.nineteeen_frame, text='Predict Resturant', command=self.predict_hd)
        self.btn_quit = tk.Button(self.nineteeen_frame, text='Quit', command=self.main_window.destroy)

        self.hd_predict_ta.pack(side='left')
        self.btn_predict.pack()
        self.btn_quit.pack()

        self.one_frame.pack()
        self.three_frame.pack()
        self.four_frame.pack()
        self.five_frame.pack()
        self.six_frame.pack()
        self.seven_frame.pack()
        self.eight_frame.pack()
        self.nine_frame.pack()
        self.ten_frame.pack()
        self.eleven_frame.pack()
        self.twelve_frame.pack()
        self.thirteen_frame.pack()
        self.fourteen_frame.pack()
        self.fifteen_frame.pack()
        self.sixteen_frame.pack()
        self.seventeen_frame.pack()
        self.eighteen_frame.pack()
        self.nineteeen_frame.pack()

        tk.mainloop()

    def predict(self, resturant, item, calories, cal_fat, total_fat, sat_fat, trans_fat, cholesterol, sodium,
                total_carb, fiber, sugar, protein, vit_a, hd):
        # Filter the data by the specified restaurant and item
        data_filtered = self.data[(self.data['RESTAURANT'] == resturant) & (self.data['ITEM'] == item)]


        # Extract the relevant nutrient information from the data
        data_filtered = data_filtered.iloc[0]
        calories_data = data_filtered['CALORIES']
        cal_fat_data = data_filtered['CALORIES FROM FAT']
        total_fat_data = data_filtered['TOTAL FAT']
        sat_fat_data = data_filtered['SATURATED FAT']
        trans_fat_data = data_filtered['TRANS FAT']
        cholesterol_data = data_filtered['CHOLESTEROL']
        sodium_data = data_filtered['SODIUM']
        total_carb_data = data_filtered['TOTAL CARBOHYDRATES']
        fiber_data = data_filtered['DIETARY FIBER']
        sugar_data = data_filtered['SUGARS']
        protein_data = data_filtered['PROTEIN']
        vit_a_data = data_filtered['VITAMIN A']

        # Calculate the nutrient values per serving
        calories_per_serving = calories / calories_data
        cal_fat_per_serving = cal_fat / cal_fat_data
        total_fat_per_serving = total_fat / total_fat_data
        sat_fat_per_serving = sat_fat / sat_fat_data
        trans_fat_per_serving = trans_fat / trans_fat_data
        cholesterol_per_serving = cholesterol / cholesterol_data
        sodium_per_serving = sodium / sodium_data
        total_carb_per_serving = total_carb / total_carb_data
        fiber_per_serving = fiber / fiber_data
        sugar_per_serving = sugar / sugar_data
        protein_per_serving = protein / protein_data
        vit_a_per_serving = vit_a / vit_a_data

        # Use the nutrient values to make a prediction
        prediction = ''
        if calories_per_serving > 1.2:
            prediction += 'High in calories. '
        if cal_fat_per_serving > 1.2:
            prediction += 'High in calories from fat. '
        if total_fat_per_serving > 1.2:
            prediction += 'High in total fat. '
        if sat_fat_per_serving > 1.2:
            prediction += 'High in saturated fat. '
        if trans_fat_per_serving > 0:
            prediction += 'Contains trans fat. '
        if cholesterol_per_serving > 1.2:
            prediction += 'High in cholesterol. '
        if sodium_per_serving > 1.2:
            prediction += 'High in sodium. '
        if total_carb_per_serving > 1.2:
            prediction += 'High in total carbohydrates. '
        if fiber_per_serving < 0.8:
            prediction += 'Low in fiber. '
        if sugar_per_serving > 1.2:
            prediction += 'High'


my_Fastfood_GUI = Fastfood_GUI