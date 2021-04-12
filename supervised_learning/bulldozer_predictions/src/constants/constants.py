
ORDINAL_COLUMNS = ['UsageBand', 'ProductSize', 'Ride_Control', 'Grouser_Type',
                   'Differential_Type', 'Steering_Controls']
OHE_COLUMNS = ['state', 'ProductGroupDesc', 'Drive_System', 'Stick',
               'Transmission', 'Track_Type']

COLUMNS_TO_REMOVE = ["MachineID", "fiModelDesc", "fiBaseModel", "fiSecondaryDesc", "fiModelSeries",
                     "fiModelDescriptor", "fiProductClassDesc", "ProductGroup", "Enclosure",
                     "Forks", "Pad_Type", "Turbocharged", "Blade_Extension", "Blade_Width", "Enclosure_Type",
                     "Engine_Horsepower", "Pushblock", "Ripper", "Scarifier", "Tip_Control", "auctioneerID",
                     "Coupler", "Coupler_System", "Grouser_Tracks", "Hydraulics_Flow", "Undercarriage_Pad_Width",
                     "Stick_Length", "Thumb", "Pattern_Changer", "Backhoe_Mounting", "Blade_Type", "Travel_Controls",
                     "Hydraulics"]