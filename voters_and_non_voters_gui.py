import tkinter as tk

class myGUI:
    def __init__(self):
        # Create the main window.
        self.main_window = tk.Tk()
        self.main_window.title("Predicting Voting Behaviours")

        # Create frames to group widgets.
        self.one_frame = tk.Frame(self.main_window)
        self.two_frame = tk.Frame(self.main_window)
        self.three_frame = tk.Frame(self.main_window)
        self.four_frame = tk.Frame(self.main_window)
        self.five_frame = tk.Frame(self.main_window)
        self.six_frame = tk.Frame(self.main_window)
        self.seven_frame = tk.Frame(self.main_window)
        self.eight_frame = tk.Frame(self.main_window)

        self.title_label = tk.Label(self.one_frame, text='VOTER INFORMATION', fg="Blue", font=("Helvetica", 18))
        self.title_label.pack()

        # Create the widgets for the frames.
        self.voter_id_label = tk.Label(self.two_frame, text='ID number:')
        self.voter_id_entry = tk.Entry(self.two_frame, bg="white", fg="black", width=10)
        self.voter_id_label.pack(side='left')
        self.voter_id_entry.pack(side='left')

        self.gender_label = tk.Label(self.three_frame, text='Gender:')
        self.click_gender_var = tk.StringVar()
        self.click_gender_var.set("")
        self.gender_inp = tk.OptionMenu(self.three_frame, self.click_gender_var, "Female", "Male")
        self.gender_label.pack(side='left')
        self.gender_inp.pack(side='left')

        self.race_label = tk.Label(self.four_frame, text='Race:')
        self.click_race_var = tk.StringVar()
        self.click_race_var.set("")
        self.race_inp = tk.OptionMenu(self.four_frame, self.click_race_var, "Black", "White", "Hispanic", "Mixed/Other")
        self.race_label.pack(side='left')
        self.race_inp.pack(side='left')

        self.education_label = tk.Label(self.five_frame, text='Education:')
        self.click_education_var = tk.StringVar()
        self.click_education_var.set("")
        self.education_inp = tk.OptionMenu(self.five_frame, self.click_education_var, "College", "High School or less",
                                           "Some college")
        self.education_label.pack(side='left')
        self.education_inp.pack(side='left')

        self.income_label = tk.Label(self.six_frame, text='Yearly Income:')
        self.click_income_var = tk.StringVar()
        self.click_income_var.set("")
        self.income_inp = tk.OptionMenu(self.six_frame, self.click_income_var, "$125k or more", "$40k-75k", "$75k-125k",
                                        "Less than $40k")
        self.income_label.pack(side='left')
        self.income_inp.pack(side='left')

        # Create predict button and quit button
        self.btn_predict = tk.Button(self.seven_frame, text='Predict', command=self.predict)
        self.btn_quit = tk.Button(self.seven_frame, text='Quit', command=self.main_window.destroy)
        self.hd_predict_ta = tk.Text(self.eight_frame, height=10, width=35, bg='light blue')

        self.hd_predict_ta.pack(side='left')
        self.btn_predict.pack()
        self.btn_quit.pack()

        self.one_frame.pack()
        self.two_frame.pack()
        self.three_frame.pack()
        self.four_frame.pack()
        self.five_frame.pack()
        self.six_frame.pack()
        self.seven_frame.pack()
        self.eight_frame.pack()

        self.main_window.mainloop()

    def predict(self):
        result_string = ""

        self.hd_predict_ta.delete(0.0, tk.END)
        voter_id = float(self.voter_id_entry.get())
        education = self.click_education_var.get()
        gender = self.click_gender_var.get()
        race = self.click_race_var.get()
        income = self.click_income_var.get()

        if gender == 'Female':
            gender = 0
        else:
            gender = 1

        if race == 'Black':
            race = 0
        elif race == 'White':
            race = 1
        elif race == 'Hispanic':
            race = 2
        else:
            race = 3

        if education == 'College':
            education = 0
        elif education == 'High School or less':
            education = 1
        else:
            education = 2

        if income == '$125k or more':
            income = 0
        elif income == '$40k-75k':
            income = 1
        elif income == '$75k-125k':
            income = 2
        else:
            income = 3

        result_string += "===Voter Prediction=== \n"
        voter_info = (voter_id, gender, race, education, income)
        voter_prediction = best_model.predict([voter_info])
        disp_string = "This prediction has an accuracy of:", str(model_accuracy)

        if voter_prediction == [0]:
            result_string += disp_string + '\n' + "This voter is likely to always vote."
        elif voter_prediction == [1]:
            result_string += disp_string + '\n' + "This voter is likely to vote rarely/never."
        else:
            result_string += disp_string + '\n' + "This voter is likely to vote sporadically."

        self.hd_predict_ta.insert('1.0', result_string)


my_gui = myGUI()
