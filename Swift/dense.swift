
struct NDarray {
    // let dtype = type(of:)
    var values: [[Int]]
    // var shape: [dtype]

    init(_ values: [[Int]]) {
        self.values = values.compactMap { $0 }
    }
}

extension NDarray: CustomStringConvertible {
    var description: String {
        return "NDArray(\(values))"
    }
}

print(NDarray([[1, [2, 3]], [4, 5, 6]]).values)
