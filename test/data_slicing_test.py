from test.test_functions import generate_testdata

data = generate_testdata()

print(data.shape)
print(data)

subset = data[20:30, 50:100]
print(subset)
print(subset.shape)

d0 = data[0]
ds = d0[20:30, 12:89]
print(ds.shape)