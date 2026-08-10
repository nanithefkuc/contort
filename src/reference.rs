use crate::{field::Field, polynomial::Polynomial};

pub(crate) fn enumerate_polynomials<F: Field>(
    field_elements: &[F],
    degree_bound: usize,
) -> Vec<Polynomial<F>> {
    fn enumerate_from<F: Field>(
        field_elements: &[F],
        coefficients: &mut [F],
        index: usize,
        output: &mut Vec<Polynomial<F>>,
    ) {
        if index == coefficients.len() {
            output.push(Polynomial::from_coefficients(coefficients.to_vec()));
            return;
        }

        for &element in field_elements {
            coefficients[index] = element;
            enumerate_from(field_elements, coefficients, index + 1, output);
        }
    }

    let mut output = Vec::new();
    let mut coefficients = vec![F::ZERO; degree_bound];
    enumerate_from(field_elements, &mut coefficients, 0, &mut output);
    output
}

pub(crate) fn hamming_distance<F: Field>(left: &[F], right: &[F]) -> Option<usize> {
    (left.len() == right.len()).then(|| {
        left.iter()
            .zip(right)
            .filter(|(left, right)| left != right)
            .count()
    })
}
