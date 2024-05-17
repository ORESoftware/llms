

template = "Hello, {0}! You have {1} new messages."

# Inject values later
name = "Alice"
count = 5
result = template.format(name, count)
print(result)  # Output: Hello, Alice! You have 5 new messages.